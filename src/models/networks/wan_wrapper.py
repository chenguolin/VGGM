# Modified from https://github.com/guandeh17/Self-Forcing/blob/main/utils/wan_wrapper.py

from typing import *
from torch import Tensor

import os
import torch
from torch import nn
import torch.nn.functional as tF
from einops import rearrange

from .wan_modules.clip import clip_xlm_roberta_vit_h_14
from .wan_modules.t5 import umt5_xxl
from .wan_modules.tokenizers import HuggingfaceTokenizer
from .wan_modules.vae import _video_vae
from .wan_modules.model import WanModel
from .wan_modules.causal_model import CausalWanModel
from .scheduler import FlowMatchScheduler


class WanCLIPEncoderWrapper(nn.Module):
    def __init__(self, pretrained_dir: str):
        super().__init__()

        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu",
        )
        self.model = self.model.eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(
            os.path.join(pretrained_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"), map_location="cpu", weights_only=False))

        self.model.textual = None  # not used
        # self.tokenizer = HuggingfaceTokenizer(
        #     name=os.path.join(pretrained_dir, "xlm-roberta-large"), seq_len=self.model.max_text_len-2, clean="whitespace")

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, videos: List[Tensor]):
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            tF.interpolate(
                u.transpose(0, 1),
                size=size,
                mode="bicubic",
                align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        return self.model.visual(videos, use_31_block=True)


class WanTextEncoderWrapper(nn.Module):
    def __init__(self, pretrained_dir: str):
        super().__init__()

        self.model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu",
        ).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(
            os.path.join(pretrained_dir, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", weights_only=False))

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(pretrained_dir, "google/umt5-xxl"), seq_len=512, clean='whitespace')  # `512` is hard-coded here

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]):
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.model(ids, mask)

        for u, v in zip(prompt_embeds, seq_lens):
            u[v:] = 0.  # set padding to 0.
        return prompt_embeds  # (B, N=512, D)


class WanVAEWrapper(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()

        self.register_buffer("mean", torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]))
        self.register_buffer("std", torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]))

        self.model = _video_vae(
            pretrained_path=pretrained_path,
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode(self, videos: Tensor):
        return torch.stack([
            self.model.encode(u.unsqueeze(0), [self.mean, 1./self.std]).float().squeeze(0)
            for u in videos
        ], dim=0)  # (B, D, f, h, w)

    def decode(self, latents: Tensor):
        return torch.stack([
            self.model.decode(latent.unsqueeze(0), [self.mean, 1./self.std]).float().clamp(-1., 1.).squeeze(0)
            for latent in latents
        ], dim=0)  # (B, 3, F, H, W)


class WanDiffusionWrapper(nn.Module):
    def __init__(self,
        pretrained_dir: str,
        concat_in_dim: int = 0,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        shift: float = 5.,
        sigma_min: float = 0.,
        extra_one_step: bool = True,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        #
        is_causal: bool = False,
        sink_size: int = 0,
        chunk_size=1,
        max_attention_size: int = 32760,  # 121 x 480 x 832 -> 35 x 30 x 52
        rope_outside: bool = False,
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                pretrained_dir,
                sink_size=sink_size,
                chunk_size=chunk_size,
                max_attention_size=max_attention_size,
                rope_outside=rope_outside,
            )
        else:
            self.model = WanModel.from_pretrained(pretrained_dir)

        self.model.use_gradient_checkpointing = use_gradient_checkpointing
        self.model.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # Handle concat inputs
        if concat_in_dim > 0:
            new_conv = nn.Conv3d(
                self.model.in_dim + concat_in_dim,
                self.model.dim,
                kernel_size=self.model.patch_size,
                stride=self.model.patch_size,
            )
            new_conv.weight.data[:, :self.model.in_dim, ...] = self.model.patch_embedding.weight.data
            new_conv.weight.data[:, self.model.in_dim:, ...] = 0.
            if self.model.patch_embedding.bias.data is not None:
                new_conv.bias.data = self.model.patch_embedding.bias.data
            self.model.patch_embedding = new_conv

        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
        )
        self.scheduler.set_timesteps(num_train_timesteps, training=True)

        self.max_seq_len = None  # not used, because we load videos in the same shape

    def forward(self,
        noisy_latents: Tensor,  # (B, D, f, h, w)
        timesteps: Tensor,  # (B,) or (B, f)
        prompt_embeds: Tensor,  # (B, N, D')
        clip_features: Optional[Tensor] = None,  # (B, N', D'')
        cond_latents: Optional[Tensor] = None,  # (B, D, f, h, w)
        add_embeds: Optional[Tensor] = None,  # (B, D, f, h, w)
        concat_embeds: Optional[Tensor] = None,  # (B, D, f, h, w)
        #
        kv_cache: Optional[List[Dict[str, Any]]] = None,
        crossattn_cache: Optional[List[Dict[str, Any]]] = None,
        current_start: Optional[int] = 0,
        #
        clean_x: Optional[Tensor] = None,
        aug_t: Optional[Tensor] = None,
    ):
        # (Optional) Concatenate extra embeds
        if concat_embeds is not None:
            noisy_latents = torch.cat([noisy_latents, concat_embeds], dim=1)

        f, h, w = noisy_latents.shape[2:]
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).repeat(1, f)  # (B, f)
        timesteps = timesteps[:, :, None, None].repeat(1, 1, h//2, w//2).flatten(1)  # (B, f*hh*ww); `//2`: hard-coded for patch embeddig
        # timesteps = torch.cat([
        #     timesteps,
        #     timesteps.new_ones((timesteps.shape[0], self.max_seq_len - timesteps.shape[1])) * timesteps[:, 0:1],
        # ], dim=1)  # (B, `self.max_seq_len`)

        if kv_cache is not None:
            model_outputs = torch.stack(self.model(
                [noisy_latent for noisy_latent in noisy_latents],
                timesteps,
                [prompt_embed for prompt_embed in prompt_embeds],
                self.max_seq_len,
                # (Optional) Image conditions
                clip_features,
                [cond_latent for cond_latent in cond_latents] if cond_latents is not None else None,
                # (Optional) Add extra embeds
                [add_embed for add_embed in add_embeds] if add_embeds is not None else None,
                #
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            ), dim=0)  # (B, D, f, h, w)
        else:
            if clean_x is not None:  # teacher forcing
                model_outputs = torch.stack(self.model(
                    [noisy_latent for noisy_latent in noisy_latents],
                    timesteps,
                    [prompt_embed for prompt_embed in prompt_embeds],
                    self.max_seq_len,
                    # (Optional) Image conditions
                    clip_features,
                    [cond_latent for cond_latent in cond_latents] if cond_latents is not None else None,
                    # (Optional) Add extra embeds
                    [add_embed for add_embed in add_embeds] if add_embeds is not None else None,
                    #
                    clean_x=clean_x,
                    aug_t=aug_t,
                ), dim=0)  # (B, D, f, h, w)
            else:
                model_outputs = torch.stack(self.model(
                    [noisy_latent for noisy_latent in noisy_latents],
                    timesteps,
                    [prompt_embed for prompt_embed in prompt_embeds],
                    self.max_seq_len,
                    # (Optional) Image conditions
                    clip_features,
                    [cond_latent for cond_latent in cond_latents] if cond_latents is not None else None,
                    # (Optional) Add extra embeds
                    [add_embed for add_embed in add_embeds] if add_embeds is not None else None,
                ), dim=0)  # (B, D, f, h, w)

        return model_outputs

    def _convert_flow_pred_to_x0(self, flow_pred: Tensor, xt: Tensor, timestep: Tensor) -> Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, D, f, h, w]
        xt: the input noisy data with shape [B, D, f, h, w]
        timestep: the timestep with shape [B] or [B, f]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        f = flow_pred.shape[2]
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).repeat(1, f)  # (B, f)
        flow_pred = rearrange(flow_pred, "b d f h w -> (b f) d h w")
        xt = rearrange(xt, "b d f h w -> (b f) d h w")
        timestep = rearrange(timestep, "b f -> (b f)")

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        x0_pred = rearrange(x0_pred, "(b f) d h w -> b d f h w", f=f)
        return x0_pred.to(original_dtype)
