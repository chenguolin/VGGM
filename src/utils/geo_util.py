from typing import *
from torch import Tensor

from math import floor, ceil
import torch
from einops import rearrange

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.structures import Pointclouds


@torch.autocast(device_type="cuda", enabled=False)
def render_pt3d_points(
    H: int,
    W: int,
    points: Tensor,    # (M, 3)
    colors: Tensor,    # (M, 3)
    C2W: Tensor,       # (f, 4, 4)
    fxfycxcy: Tensor,  # (f, 4)
    return_depth: bool = False,
):
    f = C2W.shape[0]
    cameras = setup_pt3d_cameras(H, W, C2W.float(), fxfycxcy.float())
    render_setup = setup_pt3d_renderer(cameras, (H, W))

    renderer = render_setup["renderer"]
    rasterizer = render_setup["rasterizer"]

    point_cloud = Pointclouds(points=[points.float()] * f, features=[colors.float()] * f)

    rgb = renderer(point_cloud).permute(0, 3, 1, 2)  # (f, 3, H, W)

    if not return_depth:
        return rgb  # (f, 3, H, W)
    else:
        zbuf = rasterizer(point_cloud).zbuf  # (f, H, W, K)
        depth = zbuf[..., 0]  # (f, H, W); nearest point
        depth[torch.isinf(depth)] = 0.  # background depth set to 0
        return rgb, depth  # (f, 3, H, W), (f, H, W)


def setup_pt3d_renderer(cameras: PerspectiveCameras, image_size: int | Tuple[int, int]):
    # Define the settings for rasterization and shading.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
        points_per_pixel=10,
        bin_size=0,
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    render_setup =  {
        "cameras": cameras,
        "raster_settings": raster_settings,
        "rasterizer": rasterizer,
        "renderer": renderer,
    }

    return render_setup


def setup_pt3d_cameras(H: int, W: int, C2W: Tensor, fxfycxcy: Tensor):
    device = C2W.device

    R, T = C2W[:, :3, :3], C2W[:, :3, 3:]
    R = torch.stack([-R[:, :, 0], -R[:, :, 1], R[:, :, 2]], dim=2)  # opencv/colmap -> pytorch3d
    W2C = inverse_c2w(torch.cat([R, T], dim=2))  # (f, 4, 4)
    R, T = W2C[:, :3, :3].permute(0, 2, 1), W2C[:, :3, 3]

    return PerspectiveCameras(
        focal_length=torch.cat([fxfycxcy[:, 0:1] * W, fxfycxcy[:, 1:2] * H], dim=1),
        principal_point=torch.cat([fxfycxcy[:, 2:3] * W, fxfycxcy[:, 3:4] * H], dim=1),
        in_ndc=False,
        image_size=((H, W),),
        R=R,
        T=T,
        device=device,
    )


def filter_da3_points(
    images: Tensor,    # (f, 3, H, W) in [0, 1]
    depths: Tensor,    # (f, H, W)
    confs: Tensor,     # (f, H, W)
    C2W: Tensor,       # (f, 4, 4)
    fxfycxcy: Tensor,  # (f, 4)
    *,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    conf_thresh: float = 1.05,
    conf_thresh_percentile: float = 0.4,
    ensure_thresh_percentile: float = 0.9,
    random_sample_ratio: float = -1.,
    min_num_points: int = 10000,
    max_num_points: int = 1000000,
):
    if filter_black_bg:
        confs[(images < 16/255).all(dim=1, keepdim=True)] = 1.  # black pixels
    if filter_white_bg:
        confs[(images >= 240/255).all(dim=1, keepdim=True)] = 1.  # white pixels

    lower = torch_quantile(confs, conf_thresh_percentile).item()
    upper = torch_quantile(confs, ensure_thresh_percentile).item()
    conf_thresh = min(max(conf_thresh, lower), upper)
    valid_masks = confs >= conf_thresh  # (f, H, W)

    points = unproject_depth(depths[None], C2W[None], fxfycxcy[None])[0]  # (f, 3, H, W)

    points = rearrange(points, "f c h w -> (f h w) c")
    colors = rearrange(images, "f c h w -> (f h w) c")
    valid_masks = rearrange(valid_masks, "f h w -> (f h w)")  # (M,)
    valid_points, valid_colors = points[valid_masks, :], colors[valid_masks, :]  # (M, 3), (M, 3)

    if random_sample_ratio > 0. and random_sample_ratio <= 1.:
        sample_size = min(max_num_points, max(min_num_points, int(random_sample_ratio * valid_points.shape[0])))
        rand_idxs = torch.randperm(valid_points.shape[0], device=valid_points.device)[:sample_size]
        valid_points, valid_colors = valid_points[rand_idxs, :], valid_colors[rand_idxs, :]
    return valid_points, valid_colors


def torch_quantile(
    input: Tensor,
    q: float,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: Tensor = None,
) -> Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def project_points(
    xyz_world: Tensor,
    C2W: Tensor,
    fxfycxcy: Tensor,
    H: int,
    W: int,
    znear: float = 0.001,
    zfar: float = 100.,
    margin: int = 0,
) -> Tensor:
    """Project points in 3D world coordinate to image plane and valid_mask.

    Inputs:
        - `xyz_world`: (N, 3)
        - `C2W`: (F, 4, 4)
        - `fxfycxcy`: (F, 4,)
        - `H`: image height
        - `W`: image width
        - `znear`: near plane
        - `zfar`: far plane
        - `margin`: margin for valid mask

    Outputs:
        - `depth_map`: (F, H, W); Tensor
    """
    F = C2W.shape[0]

    W2C = inverse_c2w(C2W)
    homo_xyz_world = homogenize_points(xyz_world)  # (N, 4)
    xyz_camera = homo_xyz_world @ W2C.transpose(-2, -1)  # (F, N, 4)
    xyz_camera = xyz_camera[..., :3] / xyz_camera[..., 3:]  # (F, N, 3)

    x, y, z = xyz_camera.unbind(dim=-1)  # (F, N)
    z = z.clamp(znear, zfar)
    valid_z = (z >= znear) & (z <= zfar) & torch.isfinite(z)

    fx, fy, cx, cy = fxfycxcy[:, None, :].unbind(dim=-1)  # (F, 1)
    u = (fx * x + cx * z) / z
    v = (fy * y + cy * z) / z
    u_round = (u * W).round().long()  # (F, N)
    v_round = (v * H).round().long()  # (F, N)

    valid_mask = (u_round >= margin) & (u_round <= W-1-margin) & (v_round >= margin) & (v_round <= H-1-margin) & valid_z  # (F, N)

    depth_buffer = torch.full((F, H * W,), 1e4, device=z.device, dtype=z.dtype)
    for f in range(F):
        if not valid_mask[f].any():
            continue

        u_valid = u_round[f][valid_mask[f]]  # (M_f,)
        v_valid = v_round[f][valid_mask[f]]  # (M_f,)
        z_valid = z[f][valid_mask[f]]  # (M_f,)
        z_valid = torch.where(torch.isfinite(z_valid), z_valid, torch.tensor(1e4, device=z.device, dtype=z.dtype))

        # Considering occlusion, only keep the closest points
        linear_idx = v_valid * W + u_valid  # (M_f,)
        depth_buffer[f].scatter_reduce_(dim=0, index=linear_idx, src=z_valid, reduce="amin", include_self=True)

    return depth_buffer.view(F, H, W)


def unproject_depth(
    depth_map: Tensor,
    C2W: Tensor,
    fxfycxcy: Tensor,
) -> Tensor:
    """Unproject depth map to 3D world coordinate.

    Inputs:
        - `depth_map`: (B, V, H, W)
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `xyz_world`: (B, V, 3, H, W)
    """
    device, dtype = depth_map.device, depth_map.dtype
    B, V, H, W = depth_map.shape

    depth_map = depth_map.reshape(B*V, H, W).float()
    C2W = C2W.reshape(B*V, 4, 4).float()
    K = fxfycxcy_to_intrinsics(fxfycxcy).reshape(B*V, 3, 3).float()

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
    x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
    xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
    xyz = xyz_map.view(B*V, -1, 3)

    # Get point positions in camera coordinate
    xyz = torch.bmm(xyz, inverse_k(K).transpose(1, 2))
    xyz_map = xyz.view(B*V, H, W, 3)

    # Transform pts from camera to world coordinate
    xyz_homo = homogenize_points(xyz_map)
    xyz_world = torch.bmm(C2W, xyz_homo.reshape(B*V, -1, 4).transpose(1, 2)).to(dtype)  # (B*V, 4, H*W)
    xyz_world = xyz_world[:, :3, ...] / xyz_world[:, 3:, ...]  # (B*V, 3, H*W)
    xyz_world = xyz_world.reshape(B, V, 3, H, W)
    return xyz_world


def plucker_ray(
    H: int,
    W: int,
    C2W: Tensor,
    fxfycxcy: Tensor,
    normalize_ray_d: bool = True,
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Get Plucker ray embeddings.

    Inputs:
        - `H`: image height
        - `W`: image width
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)
        - `normalize_ray_d`: whether to normalize ray direction
        w/o normalization, xyz can be obtained by `xyz = ray_o + ray_d * depth`

    Outputs:
        - `plucker`: (B, V, 6, `h`, `w`)
        - `ray_o`: (B, V, 3, `h`, `w`)
        - `ray_d`: (B, V, 3, `h`, `w`)
    """
    device, dtype = C2W.device, C2W.dtype
    B, V = C2W.shape[:2]

    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y, x = y.to(device), x.to(device)
    y = (y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / H
    x = (x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / W
    x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=2)  # (B*V, H*W, 3)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))  # (B*V, H*W, 3)
    if normalize_ray_d:
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # (B*V, H*W, 3)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)  # (B*V, H*W, 3)
    ray_o = ray_o.reshape(B, V, H, W, 3).permute(0, 1, 4, 2, 3)  # (B, V, 3, H, W)
    ray_d = ray_d.reshape(B, V, H, W, 3).permute(0, 1, 4, 2, 3)  # (B, V, 3, H, W)

    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=2).to(dtype), ray_d.to(dtype)], dim=2)

    return plucker, (ray_o, ray_d)


def inverse_c2w(C2W: Tensor) -> Tensor:
    """Compute the inverse of a batch of camera-to-world matrices in a closed form.

    Inputs:
        - `C2W`: (..., 4, 4)

    Outputs:
        - `W2C`: (..., 4, 4)
    """
    # try:
    #     return C2W.inverse()
    # except:  # some numerical issues
    R = C2W[..., :3, :3]  # (..., 3, 3)
    t = C2W[..., :3, 3]   # (..., 3)

    R_inv = R.transpose(-2, -1)  # (..., 3, 3)
    t_inv = -torch.einsum("...ij,...j->...i", R_inv, t)  # (..., 3)

    W2C = torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1)  # (..., 3, 4)
    row3 = torch.tensor([0., 0., 0., 1.], device=R.device, dtype=R.dtype).expand_as(W2C[..., 0, :])  # (..., 4)
    W2C = torch.cat([W2C, row3.unsqueeze(-2)], dim=-2)  # (..., 4, 4)
    return W2C


def inverse_k(K: Tensor) -> Tensor:
    """Compute the inverse of a batch of intrinsics matrices in a closed form.

    Inputs:
        - `K`: (..., 3, 3)

    Outputs:
        - `K_inv`: (..., 3, 3)
    """
    # try:
    #     return K.inverse()
    # except:  # some numerical issues
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    row0 = torch.stack([1. / fx, torch.zeros_like(fx), -cx / fx], dim=-1)  # (..., 3)
    row1 = torch.stack([torch.zeros_like(fy), 1. / fy, -cy / fy], dim=-1)  # (..., 3)
    row2 = torch.tensor([0., 0., 1.], device=fx.device, dtype=fx.dtype).expand_as(row0)  # (..., 3)
    K_inv = torch.stack([row0, row1, row2], dim=-2)  # (..., 3, 3)
    return K_inv


def fxfycxcy_to_intrinsics(fxfycxcy: Tensor) -> Tensor:
    """Convert a batch of fxfycxcy to intrinsics.

    Inputs:
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `intrinsics`: (B, V, 3, 3)
    """
    fx, fy, cx, cy = fxfycxcy.unbind(dim=-1)  # each is (B, V)

    row0 = torch.stack([fx, torch.zeros_like(fx), cx], dim=-1)  # (B, V, 3)
    row1 = torch.stack([torch.zeros_like(fy), fy, cy], dim=-1)  # (B, V, 3)
    row2 = torch.tensor([0., 0., 1.], device=fx.device, dtype=fx.dtype).expand(fx.shape[0], fx.shape[1], 3)  # (B, V, 3)
    intrinsics = torch.stack([row0, row1, row2], dim=-2)  # (B, V, 3, 3)
    return intrinsics


def intrinsics_to_fxfycxcy(intrinsics: Tensor) -> Tensor:
    """Convert a batch of intrinsics to fxfycxcy.

    Inputs:
        - `intrinsics`: (B, V, 3, 3)

    Outputs:
        - `fxfycxcy`: (B, V, 4)
    """
    fx = intrinsics[:, :, 0, 0]
    fy = intrinsics[:, :, 1, 1]
    cx = intrinsics[:, :, 0, 2]
    cy = intrinsics[:, :, 1, 2]
    return torch.stack([fx, fy, cx, cy], dim=-1)  # (B, V, 4)


def homogenize_points(points: Tensor) -> Tensor:
    """Homogenize a batch of 3D points: (xyz) -> (xyz1).

    Inputs:
        - `points`: (..., 2) or (..., 3)

    Outputs:
        - `points_homo`: (..., 3) or (..., 4)
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: Tensor) -> Tensor:
    """Homogenize a batch of 3D vectors: (xyz) -> (xyz0).

    Inputs:
        - `vectors`: (..., 2) or (..., 3)

    Outputs:
        - `vectors_homo`: (..., 3) or (..., 4)
    """
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)
