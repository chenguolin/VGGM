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

        if self.opt.version_action:
            with open(f"{self.root}/videos_action_caption.jsonl", "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            # `caption_data`: uid -> {whole_caption, segments: [{segment_id, start_time, end_time, action_label, caption}, ...]}
            self.caption_data = {item["filename"]: item["caption_result"] for item in data}
            if self.opt.load_global_caption:
                self.global_caption_data = {item["filename"]: item["caption_result"]["whole_caption"] for item in data}
            uids = list(self.caption_data.keys())
        elif self.opt.version_2sdiff:
            with open(f"{self.root}/valid_captions_2sdiff.jsonl", "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            self.caption_data = {item["raw_id"]: [clip["caption"] for clip in item["long_caption_lst"]] for item in data}
            if self.opt.load_global_caption:
                self.global_caption_data = {item["raw_id"]: item["org_long_caption"] for item in data}
            uids = list(self.caption_data.keys())
        elif self.opt.version_2s35w:
            with open(f"{self.root}/valid_captions_2s35w.jsonl", "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            self.caption_data = {item["raw_id"]: item["long_caption"] for item in data}
            if self.opt.load_global_caption:
                self.global_caption_data = {item["raw_id"]: item["org_long_caption"] for item in data}
            uids = list(self.caption_data.keys())
        else:
            uids = os.listdir(f"{self.root}/valid_captions")

        indices = np.random.RandomState(seed=42).permutation(len(uids))
        if self.opt.version_action or self.opt.version_2s35w or self.opt.version_2sdiff:
            if training:
                self.uids = [uids[i] for i in indices[:int(0.95 * len(uids))]]
            else:
                self.uids = [uids[i] for i in indices[int(0.95 * len(uids)):]]
        else:
            if training:
                self.uids = [uids[i].strip(".json") for i in indices[:int(0.95 * len(uids))]]
            else:
                self.uids = [uids[i].strip(".json") for i in indices[int(0.95 * len(uids)):]]

        # Filter UIDs: one `listdir` + set lookup instead of 300k `os.path.exists` calls
        existing_videos = set(os.listdir(os.path.join(self.root, "video")))
        self.uids = [uid for uid in self.uids if f"{uid}.mp4" in existing_videos]
        self.valid_idxs = list(range(len(self.uids)))

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        if self.opt.version_action:
            caption_result = self.caption_data[uid]
            segments = caption_result["segments"]
            all_captions = [seg["caption"] for seg in segments]
            all_clip_idxs = list(range(len(segments)))  # 0-based
            clip_duration, clip_base, clip_overlap = None, 0, 0  # variable-length segments
        elif self.opt.version_2s35w or self.opt.version_2sdiff:
            all_captions = self.caption_data[uid]  # List[str]: 0-based clip captions
            all_clip_idxs = list(range(len(all_captions)))  # 0-based
            clip_duration, clip_base, clip_overlap = 2, 0, 0
        else:
            with open(f"{self.root}/valid_captions/{uid}.json", "r", encoding="utf-8") as f:
                all_captions = json.load(f)  # Dict[str, str]: clip_idx -> long caption
            all_clip_idxs = sorted([int(k) for k in all_captions.keys()])  # 1-based
            clip_duration, clip_base, clip_overlap = 5, 1, 12
        dataset_source = "Internal"

        all_clip_idx_set = set(all_clip_idxs)

        # Randomly sample num_clips from [1, `opt.num_clips`], but ensure all selected clips have captions
        max_num_clips = max(1, int(self.opt.num_clips))
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
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(int(np.random.choice(self.valid_idxs)))

        # For `version_2sdiff`, always start from clip 0 because clip 0 captions have a
        # distinct introductory style vs. continuation style in later clips
        if self.opt.version_2sdiff and 0 in valid_start_clip_idxs:
            start_clip_idx = 0
        else:
            start_clip_idx = int(np.random.choice(valid_start_clip_idxs))
        clip_idxs = [start_clip_idx + i for i in range(num_clips)]  # consecutive clip indices
        if self.opt.version_action:
            prompt = [all_captions[clip_idx] for clip_idx in clip_idxs]
        elif self.opt.version_2s35w or self.opt.version_2sdiff:
            prompt = [all_captions[clip_idx] for clip_idx in clip_idxs]
        else:
            prompt = [all_captions[str(clip_idx)] for clip_idx in clip_idxs]

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        # Re-create `vr` with lower decode resolution to save CPU memory
        new_H, new_W = self.opt.input_res
        scale = max(new_H / H, new_W / W)
        if scale < 1.:
            del vr
            vr = VideoReader(str(video_path), ctx=cpu(0), width=round(W * scale), height=round(H * scale))
        # Compute `start_frame_idx` and `end_frame_idx` from selected clips
        if self.opt.version_action:
            selected_segments = [segments[ci] for ci in clip_idxs]
            start_frame_idx = int(round(selected_segments[0]["start_time"] * fps))
            end_frame_idx = int(round(selected_segments[-1]["end_time"] * fps))
        else:
            # `clip_duration`: seconds per clip; `clip_overlap`: inter-clip overlap in frames; `clip_base`: index offset
            ci_first = clip_idxs[0] - clip_base
            start_frame_idx = int(round(ci_first * clip_duration * fps)) - clip_overlap * ci_first
            end_frame_idx = start_frame_idx + num_clips * int(round(clip_duration * fps))  # may exceed video length

        # Calculate total frames based on `num_clips`
        total_frames = (self.opt.num_input_frames - 1) * num_clips + 1
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
        if (C2W.shape[0] != fxfycxcy.shape[0]) or (C2W.shape[0] != num_frames):
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(int(np.random.choice(self.valid_idxs)))
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
        if self.opt.version_action:
            for ii, clip_idx in enumerate(clip_idxs):
                seg = segments[clip_idx]
                seg_start = int(round(seg["start_time"] * fps))
                seg_end = int(round(seg["end_time"] * fps))
                num_frames_per_clip.append(len([idx for idx in input_frame_idxs if seg_start <= idx < seg_end]))
            # Assign any remaining frames (due to rounding) to the last clip
            assigned = sum(num_frames_per_clip)
            if assigned < len(input_frame_idxs):
                num_frames_per_clip[-1] += len(input_frame_idxs) - assigned
        else:
            for ii, clip_idx in enumerate(clip_idxs):
                ci = clip_idx - clip_base
                if ii == 0:
                    clip_start_frame_idx = int(round(ci * clip_duration * fps)) - clip_overlap * ci
                else:  # to avoid overlapping frames between consecutive clips, we start the next clip from the end of the previous clip
                    clip_start_frame_idx = clip_end_frame_idx
                clip_end_frame_idx = clip_start_frame_idx + int(round(clip_duration * fps))  # may exceed video length
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
        if self.opt.load_global_caption:
            return_dict["global_caption"] = self.global_caption_data[uid]  # str
        if self.opt.load_image:
            return_dict["image"] = images  # List[(F, 3, H, W)] in [0, 1]

        # Unpack the single clip (based on actual num_clips, not opt.num_clips)
        if len(prompt) == 1:
            for key in return_dict:
                if key != "uid":
                    return_dict[key] = return_dict[key][0]

        return return_dict
