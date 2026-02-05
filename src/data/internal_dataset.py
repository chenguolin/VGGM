from typing import *

import os
import re
import numpy as np
import json
from decord import VideoReader, cpu
import torch
import torchvision.transforms as tvT

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

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        with open(f"{self.root}/valid_captions/{uid}.json", "r", encoding="utf-8") as f:
            all_captions = json.load(f)  # Dict[str, str]: clip_idx -> long caption
        dataset_source = "Internal"

        # Load prompt
        clip_idx = int(np.random.choice(list(all_captions.keys())))
        prompt = all_captions[str(clip_idx)]

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        start_frame_idx = max(0, int(round((clip_idx - 1) * 5 * fps))-12)  # `5`: hard-coded for 5s-clip; `12`: hard-coded for clip-overlap
        if start_frame_idx >= num_frames:
            if uid in self.uids:
                self.uids.remove(uid)
                if len(self.uids) == 0:
                    raise ValueError("No more valid uids in InternalDataset!")
            return self.__getitem__(np.random.randint(len(self.uids)))
        input_frame_idxs = self._frame_sample(
            num_frames,
            start_frame_idx=start_frame_idx,
            end_frame_idx=start_frame_idx + int(round(5 * fps)),
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
            images = {
                idx: tvT.ToTensor()(vr[idx].asnumpy())
                for idx in input_frame_idxs
            }
            images = torch.stack([images[idx] for idx in input_frame_idxs]).float()  # (F, 3, H, W)

            # Data augmentation
            images, depths, confs, fxfycxcy = self._data_augment(images, depths, confs, fxfycxcy)
        else:
            images = None

        # Camera normalization
        C2W = self._camera_normalize(C2W)

        return_dict = {
            "uid": uid,            # str
            "prompt": prompt,      # str
            "C2W": C2W,            # (F, 4, 4)
            "fxfycxcy": fxfycxcy,  # (F, 4)
        }
        if images is not None:
            return_dict["image"] = images  # (F, 3, H, W) in [0, 1]
        return return_dict
