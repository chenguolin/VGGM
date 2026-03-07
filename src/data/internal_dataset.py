from typing import *

import os
import numpy as np
import json
from decord import VideoReader, cpu
import torch

from src.options import Options
from src.data.base_dataset import BaseDataset


class InternalDataset(BaseDataset):
    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "internal", training)

        uids = os.listdir(f"{self.root}/valid_captions")
        indices = np.random.RandomState(seed=42).permutation(len(uids))
        if training:
            self.uids = [uids[i].strip(".json") for i in indices[:int(0.95 * len(uids))]]
        else:
            self.uids = [uids[i].strip(".json") for i in indices[int(0.95 * len(uids)):]]

        # Filter out high-resolution videos using a prepared blacklist
        high_res_file = os.path.join(self.root, "high_res_uids.json")
        if os.path.exists(high_res_file):
            with open(high_res_file, "r") as f:
                high_res_uids = set(json.load(f))
            self.uids = [uid for uid in self.uids if uid not in high_res_uids]

        self.valid_idxs = list(range(len(self.uids)))

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Sample clip indices
        MAX_TRIES = 10  # TODO: make it configurable
        uid, prompt, clip_idxs = None, None, None
        for _ in range(MAX_TRIES):
            uid = self.uids[idx]
            with open(f"{self.root}/valid_captions/{uid}.json", "r", encoding="utf-8") as f:
                all_captions = json.load(f)  # Dict[str, str]: clip_idx -> long caption
            dataset_source = "Internal"

            all_clip_idxs = sorted([int(k) for k in all_captions.keys()])
            all_clip_idx_set = set(all_clip_idxs)

            # Randomly sample num_clips from [1, `opt.num_clips`], but ensure all selected clips have captions
            max_num_clips = min(max(1, int(self.opt.num_clips)), len(all_clip_idxs))
            if self.training:
                num_clips = max_num_clips if not self.opt.random_num_clips else \
                    np.random.randint(1, max_num_clips + 1)
            else:
                num_clips = max_num_clips  # fixed to `max_num_clips` for evaluation

            valid_start_clip_idxs = [
                clip_idx for clip_idx in all_clip_idxs
                if all((clip_idx + i) in all_clip_idx_set for i in range(num_clips))
            ]
            if len(valid_start_clip_idxs) == 0:
                continue

            start_clip_idx = int(np.random.choice(valid_start_clip_idxs))
            clip_idxs = [start_clip_idx + i for i in range(num_clips)]  # consecutive clip indices
            if all(clip_idx in all_clip_idx_set for clip_idx in clip_idxs):  # ensure all selected clips have captions
                prompt = [all_captions[str(clip_idx)] for clip_idx in clip_idxs]
                break

        if clip_idxs is None or prompt is None or uid is None:
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(int(np.random.choice(self.valid_idxs)))

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        # `5`: hard-coded for 5s-clip; `12`: hard-coded for clip-overlap
        start_frame_idx = int(round((clip_idxs[0] - 1) * 5 * fps)) - 12 * (clip_idxs[0] - 1)
        end_frame_idx = int(round((clip_idxs[-1] - 1) * 5 * fps)) - 12 * (clip_idxs[-1] - 1) + int(round(5 * fps))

        # Calculate total frames based on `self.opt.num_clips`
        total_frames = (self.opt.num_input_frames - 1) * self.opt.num_clips + 1
        if self.opt.is_causal:  # make sure video latents can be divided by the causal chunk size
            total_frames_latent = 1 + (total_frames - 1) // self.opt.compression_ratio[0]
            total_frames_latent = int(np.ceil(total_frames_latent / self.opt.chunk_size) * self.opt.chunk_size)
            total_frames = 1 + (total_frames_latent - 1) * self.opt.compression_ratio[0]

        input_frame_idxs = self._frame_sample(
            num_frames,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
            target_num_frames=total_frames,
        )

        depths, confs = None, None  # no depth and conf for InternalDataset

        # Load cameras (in metric scale)
        vipe_path = video_path.replace("video", "vipe").replace(".mp4", ".npz")
        vipe_data = np.load(vipe_path, allow_pickle=True)
        C2W, fxfycxcy = vipe_data["pose"], vipe_data["intrinsics"]
        assert C2W.shape[0] == fxfycxcy.shape[0] == num_frames
        C2W = torch.from_numpy(C2W).float()[input_frame_idxs, ...]  # (F, 4, 4)
        fxfycxcy = torch.from_numpy(fxfycxcy).float()[input_frame_idxs, ...]  # (F, 3, 3)
        fxfycxcy[:, 0] /= W
        fxfycxcy[:, 1] /= H
        fxfycxcy[:, 2] /= W
        fxfycxcy[:, 3] /= H

        if self.opt.load_image:
            # Load video
            frames = vr.get_batch(input_frame_idxs).asnumpy()  # (F, H, W, C) uint8
            del vr
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (F, 3, H, W)
            del frames

            # Data augmentation
            images, depths, confs, fxfycxcy = self._data_augment(images, depths, confs, fxfycxcy)
        else:
            del vr
            images = None

        # Camera normalization
        C2W = self._camera_normalize(C2W)

        # Split into clips
        num_frames_per_clip = []
        for ii, clip_idx in enumerate(clip_idxs):
            if ii == 0:
                clip_start_frame_idx = int(round((clip_idx - 1) * 5 * fps)) - 12 * (clip_idx - 1)
            else:  # to avoid overlapping frames between consecutive clips, we start the next clip from the end of the previous clip
                clip_start_frame_idx = clip_end_frame_idx
            clip_end_frame_idx = clip_start_frame_idx + int(round(5 * fps))
            num_frames_per_clip.append(len([idx for idx in input_frame_idxs if clip_start_frame_idx <= idx < clip_end_frame_idx]))
        assert sum(num_frames_per_clip) == len(input_frame_idxs) and len(num_frames_per_clip) == len(prompt)
        C2W = torch.split(C2W, num_frames_per_clip, dim=0)  # List of (F_i, 4, 4)
        fxfycxcy = torch.split(fxfycxcy, num_frames_per_clip, dim=0)  # List of (F_i, 4)
        if images is not None:
            images = torch.split(images, num_frames_per_clip, dim=0)  # List of (F_i, 3, H, W)

        return_dict = {
            "uid": uid,            # str
            "prompt": prompt,      # List[str]
            "C2W": C2W,            # List[(F, 4, 4)]
            "fxfycxcy": fxfycxcy,  # List[(F, 4)]
        }
        if self.opt.load_image:
            return_dict["image"] = images  # List[(F, 3, H, W)] in [0, 1]

        # Unpack the single clip (based on actual num_clips, not opt.num_clips)
        if len(prompt) == 1:
            for key in return_dict:
                if key != "uid":
                    return_dict[key] = return_dict[key][0]

        return return_dict
