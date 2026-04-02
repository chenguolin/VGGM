"""
Standalone DA3 (Depth-Anything-3) wrapper for frozen teacher inference.
Produces metric depth, camera extrinsics/intrinsics, and a compact pose encoding.
"""

from typing import *
from torch import Tensor

import torch
from torch import nn

from src.options import Options
from src.utils import inverse_c2w, intrinsics_to_fxfycxcy


class DA3Wrapper(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.model.da3 import NestedDepthAnything3Net

        self.opt = opt

        # Load pretrained DA3 model (nested giant-large 1.1)
        _da3 = DepthAnything3.from_pretrained(f"depth-anything/DA3NESTED-GIANT-LARGE-1.1")  # TODO: make it configurable
        self.config, self.model = _da3.config, _da3.model
        self.model: NestedDepthAnything3Net
        self.model.eval()

        del _da3

    def forward(self, images: Tensor):
        """
        Args:
            images: (B, F, 3, H, W) in [0, 1]

        Returns:
            dict with keys:
                depth:      (B, F, H, W)    metric depth
                depth_conf: (B, F, H, W)    depth confidence
                pose_enc:   (B, F, 9)       pose encoding [translation(3), quaternion(4), fov_h(1), fov_w(1)]
                C2W:        (B, F, 4, 4)    camera-to-world matrix (aligned to first frame)
                fxfycxcy:   (B, F, 4)       normalized intrinsics (fx, fy, cx, cy) / (W, H)
        """
        from depth_anything_3.model.utils.transform import mat_to_quat

        H, W = images.shape[-2:]  # (B, F, 3, H, W)

        # Run DA3 forward pass
        outputs = self.model(images)
        depths, depths_conf = outputs.depth, outputs.depth_conf  # (B, F, H, W), (B, F, H, W)
        extrinsics, intrinsics = outputs.extrinsics, outputs.intrinsics  # (B, F, 3, 4), (B, F, 3, 3)

        # Pad extrinsics from 3x4 to 4x4 by appending [0, 0, 0, 1] row
        extrinsics = torch.cat([
            extrinsics,
            torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device)
            .view(1, 1, 1, 4).repeat(extrinsics.shape[0], extrinsics.shape[1], 1, 1)
        ], dim=2)  # (B, F, 4, 4)

        # Convert W2C extrinsics to C2W, then align to first frame
        C2W = inverse_c2w(extrinsics)  # (B, F, 4, 4)
        C2W = inverse_c2w(C2W[:, 0:1, ...]) @ C2W  # align to the first frame

        # Normalize intrinsics by image dimensions
        fxfycxcy = intrinsics_to_fxfycxcy(intrinsics)  # (B, F, 4)
        fxfycxcy[:, :, 0] /= W
        fxfycxcy[:, :, 1] /= H
        fxfycxcy[:, :, 2] /= W
        fxfycxcy[:, :, 3] /= H

        # Build compact pose encoding: [translation(3), quaternion(4), fov_h(1), fov_w(1)]
        pose_enc = torch.cat([
            C2W[:, :, :3, 3].float(),                          # (B, F, 3); translation
            mat_to_quat(C2W[:, :, :3, :3].float()),            # (B, F, 4); quaternion
            2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, F, 1); fy -> fov_h
            2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, F, 1); fx -> fov_w
        ], dim=-1).to(C2W.dtype)  # (B, F, 9)

        return {
            "depth": depths,            # (B, F, H, W)
            "depth_conf": depths_conf,  # (B, F, H, W)
            "pose_enc": pose_enc,       # (B, F, 9)
            "C2W": C2W,                 # (B, F, 4, 4)
            "fxfycxcy": fxfycxcy,       # (B, F, 4)
        }
