from typing import *
from torch import Tensor

import os
import numpy as np
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.utils.geo_util import inverse_c2w
from src.data.easy_dataset import EasyDataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class BaseDataset(EasyDataset):
    def __init__(self, opt: Options, name: str, training: bool):
        super().__init__()

        self.opt = opt
        self.name = name
        self.training = training

        self.root = opt.dataset_dir_train[name] if training else opt.dataset_dir_test[name]
        if self.root.endswith("/"):
            self.root = self.root[:-1]

    def __len__(self):
        raise NotImplementedError

    # ONLY implement this function for each dataset
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples into a batch of tensors.
        """
        return_batch = {}
        for key in batch[0].keys():
            if key in ["uid", "prompt"]:
                return_batch[key] = [sample[key] for sample in batch]  # a list of str
            else:
                return_batch[key] = torch.stack([sample[key] for sample in batch])  # a Tensor
        return return_batch

    ################################ Helper Functions ################################

    def _frame_sample(self, num_frames: int, min_stride: int = 1, max_stride: int = 1, pingpong_threshold: int = -1) -> List[int]:
        frame_idxs = np.arange(num_frames, dtype=int)
        F_all, F = len(frame_idxs), \
            self.opt.num_input_frames if self.training else self.opt.num_input_frames_test

        if not self.training:
            min_gap = max_gap = max_stride * F
        else:
            min_gap, max_gap = min_stride * F, max_stride * F

        # Pick a video clip
        if F_all >= max_gap:
            gap = np.random.randint(min_gap, max_gap + 1)
            # if self.training:
            #     start_idx = np.random.randint(0, F_all - gap + 1)
            # else:
            #     start_idx = 0
            start_idx = np.random.randint(0, F_all - gap + 1)
            clip_frame_idxs = frame_idxs[start_idx:start_idx+gap]
        else:  # some samples may be too short for bounded sampling
            gap = F_all
            clip_frame_idxs = frame_idxs

        # Looping "ping-pong" sampling
        if pingpong_threshold > 0 and F_all > pingpong_threshold:
            # [0, 1, 2, ..., N-2, N-1, N-2, ..., 1]
            pingpong = np.concatenate([clip_frame_idxs, clip_frame_idxs[-2:0:-1]])
            idxs = [pingpong[i % len(pingpong)] for i in range(F)]
            return idxs

        # Uniformly sampling
        return clip_frame_idxs[np.linspace(0, gap-1, F, dtype=int)].tolist()

    def _data_augment(self, images: Tensor, depths: Optional[Tensor], confs: Optional[Tensor], fxfycxcy: Tensor):
        images, fxfycxcy = images.clone(), fxfycxcy.clone()  # not inplace
        if depths is not None:
            depths = depths.clone()
        if confs is not None:
            confs = confs.clone()

        assert images.ndim == 4  # (F, C, H, W)
        H, W = images.shape[-2:]

        new_H, new_W = self.opt.input_res
        new_H = new_H // self.opt.size_divisor * self.opt.size_divisor
        new_W = new_W // self.opt.size_divisor * self.opt.size_divisor

        # Resize and CenterCrop images
        assert self.opt.crop_resize_ratio[0] <= self.opt.crop_resize_ratio[1]
        scale_factor_max = max(new_H / (self.opt.crop_resize_ratio[0] * H), new_W / (self.opt.crop_resize_ratio[0] * W))  # to keep cropped images are not too small
        scale_factor = max(new_H / (self.opt.crop_resize_ratio[1] * H), new_W / (self.opt.crop_resize_ratio[1] * W))
        if self.training and scale_factor_max <= 1. and scale_factor <= 1.:
            scale_factor = np.random.uniform(scale_factor, scale_factor_max)
        scaled_H, scaled_W = round(H * scale_factor), round(W * scale_factor)
        # Assume we don't have to worry about changing the intrinsics based on how the images are rounded
        images = tvT.Resize((scaled_H, scaled_W), tvT.InterpolationMode.BICUBIC)(images)  # intrinsic not changed
        # Adjust the intrinsics to account for the cropping
        images = tvT.CenterCrop((new_H, new_W))(images)  # intrinsic changed
        fxfycxcy[:, 0] *= (scaled_W / new_W)
        fxfycxcy[:, 1] *= (scaled_H / new_H)

        # (Optional) Resize and CenterCrop depths
        if depths is not None:
            depths = tvT.Resize((scaled_H//self.opt.da3_down_ratio, scaled_W//self.opt.da3_down_ratio), tvT.InterpolationMode.NEAREST_EXACT)(depths)
            depths = tvT.CenterCrop((new_H//self.opt.da3_down_ratio, new_W//self.opt.da3_down_ratio))(depths)

        # (Optional) Resize and CenterCrop confs
        if confs is not None:
            confs = tvT.Resize((scaled_H//self.opt.da3_down_ratio, scaled_W//self.opt.da3_down_ratio), tvT.InterpolationMode.NEAREST_EXACT)(confs)
            confs = tvT.CenterCrop((new_H//self.opt.da3_down_ratio, new_W//self.opt.da3_down_ratio))(confs)

        return images.clamp(0., 1.), depths, confs, fxfycxcy

    def _camera_normalize(self, C2W: Tensor) -> Tensor:
        C2W = C2W.clone()  # not inplace

        if self.opt.camera_norm_type == "none":
            pass
        elif self.opt.camera_norm_type == "canonical":
            transform = inverse_c2w(C2W[0, ...])  # (4, 4)
            C2W = transform.unsqueeze(0) @ C2W  # (F, 4, 4); first camera is canonical
        else:
            raise ValueError(f"Invalid camera normalization type: [{self.opt.camera_norm_type}]")

        return C2W
