from torch import Tensor

import math
import torch


def generate_inplace_rotation_c2w(
    num_frames: int,
    device: str = "cpu",
) -> Tensor:
    """
    Camera stays at the origin and rotates 360 degrees (yaw).

    OpenCV convention:
        +X right, +Y down, +Z forward
    """
    C2W = torch.zeros((num_frames, 4, 4), device=device)
    C2W[:, 3, 3] = 1.0  # homogeneous

    angles = torch.linspace(0, 2 * math.pi, num_frames, device=device)

    for i, theta in enumerate(angles):
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Rotation around world Y axis: world <- camera
        R = torch.tensor([
            [ cos_t, 0.0, sin_t],
            [ 0.0,   1.0, 0.0  ],
            [-sin_t, 0.0, cos_t],
        ], device=device)

        C2W[i, :3, :3] = R
        C2W[i, :3, 3] = 0.  # camera at origin

    return C2W


import torch
import math

def generate_yaw_sweep_c2w(
    num_frames: int,
    yaw_deg: float = 45.,
    device: str = "cpu",
) -> Tensor:
    """
    Camera stays at origin and sweeps yaw from -yaw_deg to +yaw_deg.

    OpenCV convention:
        +X right, +Y down, +Z forward
    """
    C2W = torch.zeros((num_frames, 4, 4), device=device)
    C2W[:, 3, 3] = 1.

    # degrees -> radians
    yaw = torch.cat([
        torch.linspace(0., math.radians(yaw_deg), num_frames // 2, device=device),
        torch.linspace(math.radians(yaw_deg), 0., num_frames - num_frames // 2, device=device),
    ], dim=0)

    for i, theta in enumerate(yaw):
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # rotation around world Y axis
        R = torch.tensor([
            [ cos_t, 0., sin_t],
            [ 0.,    1., 0.   ],
            [-sin_t, 0., cos_t],
        ], device=device)

        C2W[i, :3, :3] = R
        C2W[i, :3, 3] = 0.  # stay at origin

    return C2W
