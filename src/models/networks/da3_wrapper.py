from typing import *
from torch import Tensor
from depth_anything_3.model.da3 import NestedDepthAnything3Net

import torch
from torch import nn

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options
from src.utils import inverse_c2w, intrinsics_to_fxfycxcy


class DA3Wrapper(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        _da3 = DepthAnything3.from_pretrained(f"depth-anything/DA3NESTED-GIANT-LARGE-1.1")  # TODO: make it configurable
        self.config, self.model = _da3.config, _da3.model
        self.model: NestedDepthAnything3Net
        self.model.eval()

        del _da3

    def forward(self, images: Tensor):
        H, W = images.shape[-2:]

        outputs = self.model(images)
        depths, depths_conf = outputs.depth, outputs.depth_conf  # (B, F, H, W), (B, F, H, W)
        extrinsics, intrinsics = outputs.extrinsics, outputs.intrinsics  # (B, F, 3, 4), (B, F, 3, 3)
        extrinsics = torch.cat([
            extrinsics,
            torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device)
            .view(1, 1, 1, 4).repeat(extrinsics.shape[0], extrinsics.shape[1], 1, 1)
        ], dim=2)  # (B, F, 4, 4)

        C2W = inverse_c2w(extrinsics)  # (B, F, 4, 4)
        C2W = inverse_c2w(C2W[:, 0:1, ...]) @ C2W  # align to the first frame
        fxfycxcy = intrinsics_to_fxfycxcy(intrinsics)  # (B, F, 4)
        fxfycxcy[:, :, 0] /= W
        fxfycxcy[:, :, 1] /= H
        fxfycxcy[:, :, 2] /= W
        fxfycxcy[:, :, 3] /= H
        pose_enc = torch.cat([
            C2W[:, :, :3, 3].float(),  # (B, f, 3)
            mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
            2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
            2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
        ], dim=-1).to(C2W.dtype)  # (B, f, 9)

        return {
            "depth": depths,            # (B, F, H, W)
            "depth_conf": depths_conf,  # (B, F, H, W)
            "pose_enc": pose_enc,       # (B, F, 9)
            "C2W": C2W,                 # (B, F, 4, 4)
            "fxfycxcy": fxfycxcy,       # (B, F, 4)
        }
