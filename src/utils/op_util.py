from typing import *
from torch import Tensor
from torch.nn import Module

import torch
import torch.nn.functional as tF
from einops import rearrange


def timestamp_encode(timestamps: Tensor, dim: int = 6) -> Tensor:
    """Encode scalar timestamps into sinusoidal positional embeddings.

    Uses raw timestamps in seconds (no normalization) so the encoding reflects
    absolute physical time. This allows the model to generalize from short videos
    to longer ones — e.g., `t=2.5s` always produces the same embedding regardless
    of total video duration.

    Inputs:
        - `timestamps`: (F,) tensor of frame times in seconds (first frame = 0)
        - `dim`: output dimension (default 6 to match plucker's 6 channels)

    Outputs:
        - `encoding`: (F, dim) tensor of sinusoidal positional encodings
    """
    F = timestamps.shape[0]
    device = timestamps.device
    dtype = timestamps.dtype

    # Sinusoidal positional encoding on raw seconds
    # PE(t, 2i) = sin(t * pi / 10000^(2i/d))
    # PE(t, 2i+1) = cos(t * pi / 10000^(2i/d))
    half = dim // 2
    freqs = torch.pow(10000., torch.arange(half, dtype=dtype, device=device) * 2 / dim)  # (dim//2,)
    angles = timestamps[:, None] * torch.pi / freqs[None, :]  # (F, dim//2)
    pe = torch.stack([angles.sin(), angles.cos()], dim=-1).reshape(F, -1)  # (F, dim)
    if dim % 2 == 1:  # handle odd dim
        pe = torch.cat([pe, torch.zeros(F, 1, dtype=dtype, device=device)], dim=-1)
    return pe


def mv_interpolate(x: Tensor, *args, **kwargs) -> Tensor:
    """Interpolate a multi-view tensor of shape (B, F, C, H, W)."""
    B, F, C, H, W = x.shape
    x = rearrange(x, "b f c h w -> (b f) c h w")
    x = tF.interpolate(x, *args, **kwargs)
    x = rearrange(x, "(b f) c h w -> b f c h w", b=B, f=F)
    return x


def zero_init_module(module: Module):
    for param in module.parameters():
        param.data.fill_(0.)  # set all parameters to 0


def convert_to_buffer(module: Module, persistent: bool = True):
    # Recurse over child modules
    for child in module.children():
        convert_to_buffer(child, persistent=persistent)

    # Also re-save buffers to change persistence
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def patchify(x: Tensor, patch_size: Union[int, Tuple[int, int]], tokenize: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    p1, p2 = patch_size
    if tokenize:
        return rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p1, p2=p2)
    else:
        return rearrange(x, "b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=p1, p2=p2)


def unpatchify(x: Tensor, patch_size: Union[int, Tuple[int, int]], input_size: Union[int, Tuple[int, int]], tokenize: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    (p1, p2), (h, w) = patch_size, input_size
    if tokenize:
        return rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p1, p2=p2)
    else:
        return rearrange(x, "b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=p1, p2=p2)


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has `target_dims` dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has [{x.ndim}] dims, but target_dims is [{target_dims}], which is less"
        )
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]


def to_tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    return x
