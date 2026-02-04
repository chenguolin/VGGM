from typing import *

import os
import numpy as np
import pandas as pd
import json
from decord import VideoReader, cpu
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.data.base_dataset import BaseDataset
from src.utils.geo_util import inverse_c2w, intrinsics_to_fxfycxcy, unproject_depth


class InternalDataset(BaseDataset):
    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "internal", training)

        metadata = pd.read_csv(f"{self.root}/metadata.csv")
        indices = np.random.RandomState(seed=42).permutation(len(metadata))
        if training:
            train_idxs = indices[:int(0.95 * len(metadata))]
            self.metadata = metadata.iloc[train_idxs]
        else:
            test_idxs = indices[int(0.95 * len(metadata)):]
            self.metadata = metadata.iloc[test_idxs]

        self.valid_idxs = list(range(len(self.metadata)))

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        metadata = self.metadata.iloc[idx]
        uid = metadata["org_raw_id"]
        try:
            caption = json.loads(metadata["caption_result"])  # a list of str
        except:
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(np.random.choice(self.valid_idxs))
        dataset_source = "Internal"

        if self.opt.only_static_data:
            raise NotImplementedError

        # Load prompt
        caption = caption[np.random.randint(0, len(caption) - 1)]
        clip_idx = int(float(caption["index_idx"]))
        caption_dict = json.loads(caption["caption_result"])[0]["caption"]  # 0: EN, 1: ZH
        prompt = caption_dict[np.random.choice(["long_caption", "medium_caption"])]#, "short_caption"])]

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        input_frame_idxs = self._frame_sample(
            num_frames,
            start_frame_idx=int(round((clip_idx - 1) * 5 * fps)),  # `5`: hard-coded for 5s-clip
            end_frame_idx=int(round(clip_idx * 5 * fps)),
        )

        depths, confs = None, None  # no depth and conf for InternalDataset

        # Load cameras (in metric scale)
        vipe_path = video_path.replace("video", "vipe").replace(".mp4", ".npz")
        vipe_data = np.load(vipe_path, allow_pickle=True)
        C2W, fxfycxcy = vipe_data["pose"], vipe_data["intrinsics"]
        if num_frames != C2W.shape[0] or num_frames != fxfycxcy.shape[0]:
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(np.random.choice(self.valid_idxs))
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
