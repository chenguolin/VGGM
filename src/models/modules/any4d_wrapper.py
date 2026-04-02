"""
Standalone Any4D wrapper for frozen teacher inference.
Mirrors the DA3Wrapper interface in da3_wrapper.py.
"""

from typing import *

import torch
from torch import nn, Tensor

from src.options import Options


class Any4DWrapper(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        import os
        import numpy as np
        from omegaconf import OmegaConf

        self.opt = opt

        # Register `special_float` resolver before resolving configs
        def resolve_special_float(value):
            if value == "inf":
                return np.inf
            elif value == "-inf":
                return -np.inf
            else:
                raise ValueError(f"Unknown special float value: {value}")

        if not OmegaConf.has_resolver("special_float"):
            OmegaConf.register_new_resolver("special_float", resolve_special_float)

        # Load Hydra config for Any4D
        import hydra
        from hydra import initialize_config_dir

        config_path = os.path.join(
            os.path.dirname(__file__), "../../../extensions/Any4D/configs/train.yaml"
        )
        config_path = os.path.abspath(config_path)
        config_dir = os.path.dirname(config_path)
        config_name = os.path.basename(config_path).split(".")[0]

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize_config_dir(version_base=None, config_dir=config_dir)
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[
                "machine=local",
                "model=any4d",
                "model.encoder.uses_torch_hub=false",
                "model/task=images_only",
            ],
        )

        # Build model via Any4D's model factory
        from any4d.models import init_model

        model = init_model(cfg.model.model_str, cfg.model.model_config)

        # Load checkpoint
        ckpt = torch.load(opt.any4d_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

        model.eval()
        model.requires_grad_(False)
        self.model = model

    @torch.no_grad()
    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            images: (B, F, 3, H, W) in [0, 1]

        Returns:
            dict with keys:
                pts3d:          (B, F, 3, H, W)    world-frame 3D points (metric scaled)
                pts3d_cam:      (B, F, 3, H, W)    camera-frame 3D points (metric scaled)
                depth:          (B, F, H, W)       depth along ray (metric scaled)
                ray_directions: (B, F, 3, H, W)    unit ray directions
                scene_flow:     (B, F, 3, H, W)    scene flow (metric scaled)
                cam_trans:      (B, F, 3)          camera translation (metric scaled)
                cam_quats:      (B, F, 4)          camera quaternion (x, y, z, w)
                scale:          (B, F, 1)          metric scaling factor
                C2W:            (B, F, 4, 4)       camera-to-world matrix
                fxfycxcy:       (B, F, 4)          normalized intrinsics (recovered from ray directions)
                pose_enc:       (B, F, 9)          DA3-compatible pose encoding
                depth_conf:     (B, F, H, W)       depth confidence (continuous, ≥1)
                non_ambiguous_mask: (B, F, H, W)   binary mask (True = reliable prediction)
        """
        from any4d.utils.geometry import quaternion_to_rotation_matrix

        B, F, C, H, W = images.shape

        # DINOv2 normalization (images are [0, 1])
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
        images = (images - mean) / std

        # Build views list for Any4D (each frame is a separate view)
        views = []
        for f_idx in range(F):
            views.append({
                "img": images[:, f_idx],  # (B, 3, H, W)
                "data_norm_type": ["dinov2"],
            })

        # Run Any4D forward pass
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = self.model(views)  # List[dict], one per view

        # Stack per-view outputs along frame dimension -> (B, F, ...)
        # Any4D outputs spatial tensors as (B, H, W, C); permute to (B, C, H, W) first
        def _stack(key, spatial=False, squeeze_last=False):
            tensors = []
            for pred in preds:
                if key in pred:
                    t = pred[key]
                    if spatial and t.dim() == 4:
                        t = t.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    if squeeze_last:
                        t = t.squeeze(-1)
                    tensors.append(t)
                else:
                    return None
            return torch.stack(tensors, dim=1)  # (B, F, ...)

        pts3d = _stack("pts3d", spatial=True)                   # (B, F, 3, H, W)
        pts3d_cam = _stack("pts3d_cam", spatial=True)           # (B, F, 3, H, W)
        depth = _stack("depth_along_ray", squeeze_last=True)    # (B, F, H, W)
        ray_directions = _stack("ray_directions", spatial=True) # (B, F, 3, H, W)
        scene_flow = _stack("scene_flow", spatial=True)         # (B, F, 3, H, W)
        cam_trans = _stack("cam_trans")                         # (B, F, 3)
        cam_quats = _stack("cam_quats")                         # (B, F, 4)
        scale = _stack("metric_scaling_factor")                 # (B, F, 1)

        # Confidence and mask (guaranteed by our adaptor config `raydirs_depth_pose_confidence_mask_scale`)
        conf = _stack("conf")                                   # (B, F, H, W)
        non_ambiguous_mask = _stack("non_ambiguous_mask")       # (B, F, H, W)

        # Build C2W from `cam_trans` + `cam_quats`
        # `cam_quats` is (x, y, z, w) notation
        cam_quats_flat = cam_quats.reshape(B * F, 4)         # (B*F, 4)
        rot = quaternion_to_rotation_matrix(cam_quats_flat)  # (B*F, 3, 3)
        rot = rot.reshape(B, F, 3, 3)

        C2W = torch.zeros(B, F, 4, 4, device=images.device, dtype=rot.dtype)
        C2W[:, :, :3, :3] = rot
        C2W[:, :, :3, 3] = cam_trans
        C2W[:, :, 3, 3] = 1.

        # Align to first frame (same convention as DA3Wrapper)
        C2W_inv0 = torch.zeros_like(C2W[:, 0:1])  # (B, 1, 4, 4)
        C2W_inv0[:, :, :3, :3] = C2W[:, 0:1, :3, :3].transpose(-1, -2)
        C2W_inv0[:, :, :3, 3] = -torch.einsum(
            "bfij,bfj->bfi", C2W[:, 0:1, :3, :3].transpose(-1, -2), C2W[:, 0:1, :3, 3]
        )
        C2W_inv0[:, :, 3, 3] = 1.
        C2W = C2W_inv0 @ C2W  # align to first frame

        # Recover intrinsics from `ray_directions` via linear regression
        # `recover_pinhole_intrinsics_from_ray_directions` expects (B, H, W, 3) in pixel space
        from any4d.utils.geometry import recover_pinhole_intrinsics_from_ray_directions

        ray_dirs_hwc = ray_directions.permute(0, 1, 3, 4, 2)  # (B, F, H, W, 3)
        ray_dirs_flat = ray_dirs_hwc.reshape(B * F, H, W, 3)  # (B*F, H, W, 3)
        intrinsics = recover_pinhole_intrinsics_from_ray_directions(ray_dirs_flat)  # (B*F, 3, 3)
        intrinsics = intrinsics.reshape(B, F, 3, 3)

        # Extract and normalize fxfycxcy by image dimensions
        from src.utils import intrinsics_to_fxfycxcy
        fxfycxcy = intrinsics_to_fxfycxcy(intrinsics)  # (B, F, 4)
        fxfycxcy[:, :, 0] /= W  # fx
        fxfycxcy[:, :, 1] /= H  # fy
        fxfycxcy[:, :, 2] /= W  # cx
        fxfycxcy[:, :, 3] /= H  # cy

        # Build DA3-compatible `pose_enc`: [translation(3), quaternion(4), fov_h(1), fov_w(1)]
        from depth_anything_3.model.utils.transform import mat_to_quat

        pose_quat = mat_to_quat(C2W[:, :, :3, :3].float())        # (B, F, 4)
        pose_trans = C2W[:, :, :3, 3].float()                     # (B, F, 3)
        fov_h = 2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2]))  # (B, F, 1)
        fov_w = 2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1]))  # (B, F, 1)
        pose_enc = torch.cat([pose_trans, pose_quat, fov_h, fov_w], dim=-1).to(C2W.dtype)  # (B, F, 9)

        result = {
            "pts3d": pts3d,                   # (B, F, 3, H, W)
            "pts3d_cam": pts3d_cam,           # (B, F, 3, H, W)
            "depth": depth,                   # (B, F, H, W)
            "ray_directions": ray_directions, # (B, F, 3, H, W)
            "scene_flow": scene_flow,         # (B, F, 3, H, W)
            "cam_trans": cam_trans,           # (B, F, 3)
            "cam_quats": cam_quats,           # (B, F, 4)
            "scale": scale,                   # (B, F, 1)
            "C2W": C2W,                       # (B, F, 4, 4)
            "fxfycxcy": fxfycxcy,             # (B, F, 4)
            "pose_enc": pose_enc,             # (B, F, 9)
            "depth_conf": conf,               # (B, F, H, W)
            "non_ambiguous_mask": non_ambiguous_mask,  # (B, F, H, W)
        }

        return result
