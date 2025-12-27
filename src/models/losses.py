from typing import *
from torch import Tensor, BoolTensor

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
        f, h, w = pred_xyzs.shape[1], pred_xyzs.shape[3], pred_xyzs.shape[4]

        if masks is None:
            masks = torch.ones_like(pred_xyzs[:, :, 0, :, :]).to(torch.bool)  # (B, F, H, W)
        if confs is None:
            confs = torch.ones_like(pred_xyzs[:, :, 0, :, :])  # (B, F, H, W)

        pred_xyzs = rearrange(pred_xyzs, "b f c h w -> b (f h w) c")  # (B, N, 3)
        gt_xyzs = rearrange(gt_xyzs, "b f c h w -> b (f h w) c")  # (B, N, 3)
        confs = rearrange(confs, "b f h w -> b (f h w)")  # (B, N)

        # Confidence-weighted MSE
        xyz_loss = tF.mse_loss(pred_xyzs, gt_xyzs, reduction="none").mean(dim=-1)  # (B, N)
        if self.opt.conf_alpha > 0.:
            xyz_loss = (confs * xyz_loss - self.opt.conf_alpha * torch.log(confs))  # (B, N)

        # Filter by masks
        masks = rearrange(masks, "b f h w -> b f (h w)")  # (B, F, HW)
        xyz_loss = rearrange(xyz_loss, "b (f h w) -> b f (h w)", f=f, h=h, w=w)  # (B, F, HW)
        masks = masks.float() * (xyz_loss <= self.opt.xyz_loss_threshold).float()
        xyz_loss = (xyz_loss * masks).sum(dim=-1) / (masks.sum(dim=-1) + 1e-6)  # (B, F)
        return xyz_loss  # (B, F)


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
        f, h, w = pred_depths.shape[1], pred_depths.shape[2], pred_depths.shape[3]

        if masks is None:
            masks = torch.ones_like(pred_depths[:, :, :, :]).to(torch.bool)  # (B, F, H, W)
        if confs is None:
            confs = torch.ones_like(pred_depths[:, :, :, :])  # (B, F, H, W)

        pred_depths = rearrange(pred_depths, "b f h w -> b (f h w)")  # (B, N)
        gt_depths = rearrange(gt_depths, "b f h w -> b (f h w)")  # (B, N)
        confs = rearrange(confs, "b f h w -> b (f h w)")  # (B, N)

        # Confidence-weighted MSE
        depth_loss = tF.mse_loss(pred_depths, gt_depths, reduction="none")  # (B, N)
        if self.opt.conf_alpha > 0.:
            depth_loss = (confs * depth_loss - self.opt.conf_alpha * torch.log(confs))  # (B, N)

        # Filter by masks
        masks = rearrange(masks, "b f h w -> b f (h w)")  # (B, F, HW)
        depth_loss = rearrange(depth_loss, "b (f h w) -> b f (h w)", f=f, h=h, w=w)  # (B, F, HW)
        masks_ = masks.float() * (depth_loss <= self.opt.depth_loss_threshold).float()
        depth_loss = (depth_loss * masks_).sum(dim=-1) / (masks_.sum(dim=-1) + 1e-6)  # (B, F)

        # Gradient loss
        pred_depths = rearrange(pred_depths, "b (f h w) -> b f h w", f=f, h=h, w=w)
        gt_depths = rearrange(gt_depths, "b (f h w) -> b f h w", f=f, h=h, w=w)
        masks = rearrange(masks, "b f (h w) -> b f h w", f=f, h=h, w=w)
        confs = rearrange(confs, "b (f h w) -> b f h w", f=f, h=h, w=w)
        depth_grad_loss = self.grad_loss(pred_depths, gt_depths, masks, confs).mean()  # (B, F)

        return depth_loss + depth_grad_loss  # (B, F)


class CameraLoss(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

    def forward(self, pred_pose_encs: Tensor, gt_pose_encs: Tensor):
        camera_loss = tF.l1_loss(pred_pose_encs, gt_pose_encs, reduction="none").mean(dim=-1)  # (B, F)
        masks = (camera_loss <= self.opt.camera_loss_threshold).float()  # (B, F)
        camera_loss = (camera_loss * masks) / (masks + 1e-6)  # (B, F)
        return camera_loss  # (B, F)


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
        summed_mask = torch.sum(mask, (-2, -1))
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

        image_loss = (torch.sum(grad_x, (-2, -1)) + torch.sum(grad_y, (-2, -1))) / 2.
        image_loss = image_loss / (summed_mask + 1e-6)

        return image_loss  # (B, F)
