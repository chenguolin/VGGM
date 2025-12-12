from typing import *
from torch import Tensor
from depth_anything_3.model.da3 import DepthAnything3Net

import torch
from torch import nn
from einops import rearrange

from src.options import Options

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import quat_to_mat
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


class DA3Wrapper(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        _da3 = DepthAnything3.from_pretrained(f"depth-anything/{(self.opt.model_name.upper())}")
        self.config, self.model = _da3.config, _da3.model
        self.model: DepthAnything3Net
        self.model.eval()

        del _da3

    def forward(self, images: Tensor):
        H, W = images.shape[-2:]

        # Depth & Raymap
        feats, _ = self.model.backbone(images)
        head_outputs = self.model.head(feats, H, W, patch_start_idx=0, chunk_size=self.opt.da3_chunk_size)
        depths, depths_conf = head_outputs["depth"], head_outputs["depth_conf"]  # (B, F, H, W), (B, F, H, W)
        rays, rays_conf = head_outputs["ray"], head_outputs["ray_conf"]  # (B, F, H, W, 6), (B, F, H, W)

        # Camera
        pose_enc = self.model.cam_dec(feats[-1][1])  # (B, F, 9)
        with torch.no_grad():
            ## Camera decoder
            if not self.opt.use_ray_pose:
                R, T = quat_to_mat(pose_enc[..., 3:7]), pose_enc[..., :3]
                C2W = torch.cat([R, T[..., None]], dim=-1)  # (B, F, 3, 4)
                C2W = torch.cat([C2W, torch.zeros_like(C2W[..., :1, :])], dim=-2)  # (B, F, 4, 4)
                C2W[..., 3, 3] = 1.  # (B, F, 4, 4)

                fov_h, fov_w = pose_enc[..., 7], pose_enc[..., 8]
                fx, fy = 0.5 / torch.clamp(torch.tan(fov_w / 2.), 1e-6), 0.5 / torch.clamp(torch.tan(fov_h / 2.), 1e-6)
                cx, cy = 0.5 * torch.ones_like(fov_h), 0.5 * torch.ones_like(fov_w)
                fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1)  # (B, F, 4)
            ## Raymap
            else:
                pred_extrinsic, pred_focal_lengths, pred_principal_points = \
                    get_extrinsic_from_camray(rays, rays_conf, rays.shape[-3], rays.shape[-2])

                C2W = pred_extrinsic  # (B, F, 4, 4)
                fxfycxcy = torch.stack([
                    pred_focal_lengths[:, :, 0] / 2,
                    pred_focal_lengths[:, :, 1] / 2,
                    pred_principal_points[:, :, 0] / 2,
                    pred_principal_points[:, :, 1] / 2,
                ], dim=-1)  # (B, F, 4)

        rays = rearrange(rays, "b f h w c -> b f c h w")  # (B, F, 6, H, W)
        return {
            "depth": depths,            # (B, F, H, W)
            "depth_conf": depths_conf,  # (B, F, H, W)
            "ray": rays,                # (B, F, 6, H/2, W/2)
            "ray_conf": rays_conf,      # (B, F, H/2, W/2)
            "pose_enc": pose_enc,       # (B, F, 9)
            #
            "C2W": C2W,                 # (B, F, 4, 4)
            "fxfycxcy": fxfycxcy,       # (B, F, 4)
        }
