from typing import *
from torch import Tensor, BoolTensor

from math import floor, ceil
import torch
from torch import nn
import torch.nn.functional as tF
from einops import rearrange

from src.options import Options


class XYZLoss(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

    def forward(self,
        pred_xyzs: Tensor,
        gt_xyzs: Tensor,
        masks: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
    ):
        if masks is None:
            masks = torch.ones_like(pred_xyzs[:, :, 0, :, :]).to(torch.bool)  # (B, F, H, W)
        if confs is None:
            confs = torch.ones_like(pred_xyzs[:, :, 0, :, :])  # (B, F, H, W)

        pred_xyzs = rearrange(pred_xyzs, "b f c h w -> b (f h w) c")  # (B, N, 3)
        gt_xyzs = rearrange(gt_xyzs, "b f c h w -> b (f h w) c")  # (B, N, 3)
        masks = rearrange(masks, "b f h w -> b (f h w)")  # (B, N)
        confs = rearrange(confs, "b f h w -> b (f h w)")  # (B, N)

        # Confidence-weighted MSE
        xyz_loss = tF.mse_loss(pred_xyzs, gt_xyzs, reduction="none")  # (B, N, 3)
        if self.opt.conf_alpha > 0.:
            xyz_loss = (confs.unsqueeze(-1) * xyz_loss - self.opt.conf_alpha * torch.log(confs.unsqueeze(-1)))  # (B, N, 3)
        xyz_loss = filter_by_quantile(xyz_loss.mean(dim=-1)[masks], self.opt.filter_by_quantile).mean()  # (,)
        return xyz_loss  # (,)


class DepthLoss(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt
        self.grad_loss = GradientLoss(
            scales=opt.gradient_loss_scale, conf_alpha=opt.conf_alpha)

    def forward(self,
        pred_depths: Tensor,
        gt_depths: Tensor,
        masks: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
    ):
        if masks is None:
            masks = torch.ones_like(pred_depths[:, :, :, :]).to(torch.bool)  # (B, F, H, W)
        if confs is None:
            confs = torch.ones_like(pred_depths[:, :, :, :])  # (B, F, H, W)

        F, H, W = masks.shape[1:]

        pred_depths = rearrange(pred_depths, "b f h w -> b (f h w)")  # (B, N)
        gt_depths = rearrange(gt_depths, "b f h w -> b (f h w)")  # (B, N)
        masks = rearrange(masks, "b f h w -> b (f h w)")  # (B, N)
        confs = rearrange(confs, "b f h w -> b (f h w)")  # (B, N)

        # Confidence-weighted MSE
        depth_loss = tF.mse_loss(pred_depths, gt_depths, reduction="none")  # (B, N)
        if self.opt.conf_alpha > 0.:
            depth_loss = (confs * depth_loss - self.opt.conf_alpha * torch.log(confs))  # (B, N)
        depth_loss = filter_by_quantile(depth_loss[masks], self.opt.filter_by_quantile).mean()  # (,)

        # Gradient loss
        pred_depths = rearrange(pred_depths, "b (f h w) -> b f h w", f=F, h=H, w=W)
        gt_depths = rearrange(gt_depths, "b (f h w) -> b f h w", f=F, h=H, w=W)
        masks = rearrange(masks, "b (f h w) -> b f h w", f=F, h=H, w=W)
        confs = rearrange(confs, "b (f h w) -> b f h w", f=F, h=H, w=W)
        depth_grad_loss = self.grad_loss(pred_depths, gt_depths, masks, confs).mean()  # (,)

        return depth_loss + depth_grad_loss  # (,)


class CameraLoss(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

    def forward(self, pred_pose_encs: Tensor, gt_pose_encs: Tensor):
        camera_loss = tF.l1_loss(pred_pose_encs, gt_pose_encs, reduction="none").mean(dim=(1, 2)).mean()  # (,)
        return camera_loss  # (,)


################################ Gradient Loss ################################


# Copied from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/model_components/losses.py
# Modified:
    # (1) multi-view view: (B, H, W) -> (B, V, H, W)
    # (2) enable confidence-awareness; cf. DuSt3R, VGGT
    # (3) no `reduction_type`, so return batch-wise loss
    # (4) average loss over different scales


# losses based on https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py
class GradientLoss(nn.Module):
    """
    multiscale, scale-invariant gradient matching term to the disparity space.
    This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
    More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
    """

    def __init__(self, scales: int = 4, conf_alpha: float = 0.2):
        """
        Args:
            scales: number of scales to use
        """
        super().__init__()

        self.scales = scales
        self.conf_alpha = conf_alpha

    def forward(
        self,
        prediction: Tensor,             # (B, V, H, W)
        target: Tensor,                 # (B, V, H, W)
        mask: BoolTensor,               # (B, V, H, W)
        conf: Optional[Tensor] = None,  # (B, V, H, W)
    ) -> Tensor:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            gradient loss based on reduction function
        """
        assert self.scales >= 1
        total = 0.

        for scale in range(self.scales):
            step = pow(2, scale)

            grad_loss = self.gradient_loss(
                prediction[:, :, ::step, ::step],
                target[:, :, ::step, ::step],
                mask[:, :, ::step, ::step],
                conf[:, :, ::step, ::step] if conf is not None else None,
            )
            total += grad_loss
        total /= self.scales

        assert isinstance(total, Tensor)
        return total

    def gradient_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: BoolTensor,
        conf: Optional[Tensor] = None,
    ) -> Tensor:
        """
        multiscale, scale-invariant gradient matching term to the disparity space.
        This term biases discontinuities to be sharp and to coincide with discontinuities in the ground truth
        More info here https://arxiv.org/pdf/1907.01341.pdf Equation 11
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            reduction: reduction function, either reduction_batch_based or reduction_image_based
        Returns:
            gradient loss based on reduction function
        """
        summed_mask = torch.sum(mask, (-3, -2, -1))
        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
        mask_x = torch.mul(mask[:, :, :, 1:], mask[:, :, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
        mask_y = torch.mul(mask[:, :, 1:, :], mask[:, :, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        if conf is not None and self.conf_alpha > 0.:
            conf_x = torch.mean(conf[:, :, :, 1:] + conf[:, :, :, :-1])
            grad_x = conf_x * grad_x - self.conf_alpha * torch.log(conf_x)

            conf_y = torch.mean(conf[:, :, 1:, :] + conf[:, :, :-1, :])
            grad_y = conf_y * grad_y - self.conf_alpha * torch.log(conf_y)

        image_loss = (torch.sum(grad_x, (-3, -2, -1)) + torch.sum(grad_y, (-3, -2, -1))) / 2.
        image_loss = image_loss / (summed_mask + 1e-6)

        return image_loss  # (B,)


################################ Quantile Functions ################################


# Copied from https://github.com/facebookresearch/vggt/blob/main/training/loss.py
def filter_by_quantile(loss_tensor: Tensor, valid_range: float, min_elements=1000, hard_max=100) -> Tensor:
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.

    This helps remove outliers that could destabilize training.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input: Tensor,
    q: float,
    dim: Optional[int] = None,
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
