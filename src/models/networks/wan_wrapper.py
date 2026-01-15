# Modified from https://github.com/guandeh17/Self-Forcing/blob/main/utils/wan_wrapper.py

from typing import *
from torch import Tensor
from depth_anything_3.model.da3 import DepthAnything3Net

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

from .wan_modules.model import sinusoidal_embedding_1d
from src.utils import zero_init_module, inverse_c2w, fxfycxcy_to_intrinsics

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import quat_to_mat
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray


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
        ], dim=0)  # (B, 3, f, H, W)


class WanDiffusionWrapper(nn.Module):
    def __init__(self,
        pretrained_dir: str,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        shift: float = 5.,
        sigma_min: float = 0.,
        extra_one_step: bool = True,
        #
        input_plucker: bool = False,
        extra_condition_dim: int = 0,
        #
        memory_num_tokens: int = 0,
        #
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        #
        is_causal: bool = False,
        sink_size: int = 0,
        chunk_size=1,
        max_attention_size: int = 32760,  # 81 x 480 x 832 -> 21 x 30 x 52
        rope_outside: bool = False,
        **kwargs,  # for compatibility
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

        # (Optional) Plucker embedding
        self.input_plucker = input_plucker
        if input_plucker:
            self.plucker_embed = nn.Conv2d(6, self.model.dim, kernel_size=16, stride=16)  # `16`: hard-coded for Wan2.1
            zero_init_module(self.plucker_embed)

        # (Optional) Extra condition
        self.extra_condition_dim = extra_condition_dim
        if extra_condition_dim > 0:
            self.extra_condition_embed = nn.Conv2d(extra_condition_dim, self.model.dim, kernel_size=16, stride=16)  # `16`: hard-coded for Wan2.1
            zero_init_module(self.extra_condition_embed)

        # (Optional) Memory module
        if memory_num_tokens > 0:
            self.init_state = nn.Embedding(memory_num_tokens, self.model.dim)
            self.init_state.weight.data.normal_(0., 0.02)
        self.memory_num_tokens = memory_num_tokens

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
        #
        plucker: Optional[Tensor] = None,  # (B, f, 6, H, W)
        extra_condition: Optional[Tensor] = None,  # (B, f, C, H, W)
        C2W: Optional[Tensor] = None,  # (B, f, 4, 4)
        fxfycxcy: Optional[Tensor] = None,  # (B, f, 4)
        #
        kv_cache: Optional[List[Dict[str, Any]]] = None,
        crossattn_cache: Optional[List[Dict[str, Any]]] = None,
        current_start: Optional[int] = 0,
        #
        memory_tokens: Optional[Tensor] = None,
        #
        rolling: bool = False,
        update_cache: bool = False,
        chunk_size: int = 1,
        #
        kv_cache_da3: Optional[List[Dict[str, Any]]] = None,  # not used; for compatibility
        current_start_da3: Optional[int] = 0,  # not used; for compatibility
        #
        clean_x: Optional[Tensor] = None,
        aug_t: Optional[Tensor] = None,
    ):
        f, h, w = noisy_latents.shape[2:]
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).repeat(1, f)  # (B, f)
        timesteps = timesteps[:, :, None, None].repeat(1, 1, h//2, w//2).flatten(1)  # (B, f*hh*ww); `//2`: hard-coded for patch embeddig

        # (Optional) Plucker embedding
        if self.input_plucker:
            plucker = rearrange(plucker, "b f c h w -> (b f) c h w").to(noisy_latents.dtype)
            plucker_embeds = self.plucker_embed(plucker)
            plucker_embeds = rearrange(plucker_embeds, "(b f) c h w -> b c f h w", f=f)  # (B, D, f, hh, ww)
        else:
            plucker_embeds = None

        # (Optional) Extra condition embedding
        if self.extra_condition_dim > 0:
            extra_condition = rearrange(extra_condition, "b f c h w -> (b f) c h w").to(noisy_latents.dtype)
            extra_condition_embeds = self.extra_condition_embed(extra_condition)
            extra_condition_embeds = rearrange(extra_condition_embeds, "(b f) c h w -> b c f h w", f=f)  # (B, D, f, hh, ww)
            if plucker is not None:
                plucker_embeds = plucker_embeds + extra_condition_embeds
            else:
                plucker_embeds = extra_condition_embeds
        else:
            extra_condition_embeds = None

        if kv_cache is not None:
            model_outputs = self.model(
                [noisy_latent for noisy_latent in noisy_latents],
                timesteps,
                [prompt_embed for prompt_embed in prompt_embeds],
                self.max_seq_len,
                # (Optional) Image conditions
                clip_features,
                [cond_latent for cond_latent in cond_latents] if cond_latents is not None else None,
                # (Optional) Add extra embeds
                [plucker_embed for plucker_embed in plucker_embeds] if plucker_embeds is not None else None,
                #
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                #
                memory_tokens=memory_tokens,
                #
                rolling=rolling,
                update_cache=update_cache,
                chunk_size=chunk_size,
            )
            if memory_tokens is not None:
                model_outputs, memory_tokens = model_outputs
            model_outputs = torch.stack(model_outputs, dim=0)  # (B, D, f, h, w)

        else:
            assert memory_tokens is None

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
                    [plucker_embed for plucker_embed in plucker_embeds] if plucker_embeds is not None else None,
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
                    [plucker_embed for plucker_embed in plucker_embeds] if plucker_embeds is not None else None,
                ), dim=0)  # (B, D, f, h, w)

        if memory_tokens is not None:
            return model_outputs, memory_tokens
        else:
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
        sigma_t = sigmas[timestep_id]

        # NOTE: handle (B*f,) input shape and timestep=0 for I2V
        sigma_t[timestep == 0.] = 0.
        sigma_t = sigma_t.reshape(-1, 1, 1, 1)

        x0_pred = xt - sigma_t * flow_pred
        x0_pred = rearrange(x0_pred, "(b f) d h w -> b d f h w", f=f)
        return x0_pred.to(original_dtype)


class WanDiffusionDA3Wrapper(nn.Module):
    def __init__(self,
        pretrained_dir: str,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        shift: float = 5.,
        sigma_min: float = 0.,
        extra_one_step: bool = True,
        #
        input_plucker: bool = False,
        extra_condition_dim: int = 0,
        #
        memory_num_tokens: int = 0,
        #
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        #
        is_causal: bool = False,
        sink_size: int = 0,
        chunk_size=1,
        max_attention_size: int = 32760,  # 81 x 480 x 832 -> 21 x 30 x 52
        rope_outside: bool = False,
        #
        da3_model_name: str = "da3-large-1.1",
        da3_chunk_size: int = 8,
        da3_down_ratio: int = 1,
        da3_use_ray_pose: bool = False,
        da3_interactive: bool = False,
        da3_input_cam: bool = True,
        da3_max_attention_size: int = 32781,  # 81 x 480 x 832 -> 21 x (30 x 52 + 1), +1 for camera token
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

        # (Optional) Plucker embedding
        self.input_plucker = input_plucker
        if input_plucker:
            self.plucker_embed = nn.Conv2d(6, self.model.dim, kernel_size=16, stride=16)  # `16`: hard-coded for Wan2.1
            zero_init_module(self.plucker_embed)

        # (Optional) Extra condition
        self.extra_condition_dim = extra_condition_dim
        if extra_condition_dim > 0:
            self.extra_condition_embed = nn.Conv2d(extra_condition_dim, self.model.dim, kernel_size=16, stride=16)  # `16`: hard-coded for Wan2.1
            zero_init_module(self.extra_condition_embed)

        # (Optional) Memory module
        if memory_num_tokens > 0:
            self.init_state = nn.Embedding(memory_num_tokens, self.model.dim)
            self.init_state.weight.data.normal_(0., 0.02)
        self.memory_num_tokens = memory_num_tokens

        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            shift=shift,
            sigma_min=sigma_min,
            extra_one_step=extra_one_step,
        )
        self.scheduler.set_timesteps(num_train_timesteps, training=True)

        self.max_seq_len = None  # not used, because we load videos in the same shape

        # Load DA3
        assert da3_model_name == "da3-large-1.1", "By now, only `da3-large-1.1` is supported"
        _da3 = DepthAnything3.from_pretrained(f"depth-anything/{(da3_model_name.upper())}")
        self.da3_model: DepthAnything3Net = _da3.model
        self.da3_model.backbone.pretrained.patch_size = 16  # hard-coded for Wan2.1
        self.da3_model.head.patch_size = 16  # hard-coded for Wan2.1
            ## Remove not used modules
        if not da3_input_cam:
            self.da3_model.cam_enc = None
        else:
            self.da3_model.backbone.pretrained.camera_token = None
        self.da3_model.backbone.pretrained.patch_embed = None

        del _da3

        # Extra modules of WanDA3
        self.da3_adapter = nn.Linear(1536, 1024)  # hard-coded for Wan2.1-1.3B to DA3-large
        if da3_interactive:
            self.dit_da3_interactive = nn.ModuleList([
                # `1536` and `1024` are hard-coded for Wan1.3B and DA3-large
                InteractiveModule(dim1=1536, dim2=1024, dim=1536, num_heads=24, down_ratio=da3_down_ratio)
                for _ in range(24)  # `24`: hard-coded for DA3-large
            ])

        self.da3_chunk_size = da3_chunk_size
        self.da3_down_ratio = da3_down_ratio
        self.da3_use_ray_pose = da3_use_ray_pose
        self.da3_interactive = da3_interactive
        self.da3_input_cam = da3_input_cam
        self.da3_max_attention_size = da3_max_attention_size

        self.is_causal = is_causal
        self.block_mask = None

    def forward(self,
        noisy_latents: Tensor,  # (B, D, f, h, w)
        timesteps: Tensor,  # (B,) or (B, f)
        prompt_embeds: Tensor,  # (B, N, D')
        clip_features: Optional[Tensor] = None,  # (B, N', D'')
        cond_latents: Optional[Tensor] = None,  # (B, D, f, h, w)
        #
        plucker: Optional[Tensor] = None,  # (B, f, 6, H, W)
        extra_condition: Optional[Tensor] = None,  # (B, f, C, H, W)
        C2W: Optional[Tensor] = None,  # (B, f, 4, 4)
        fxfycxcy: Optional[Tensor] = None,  # (B, f, 4)
        #
        kv_cache: Optional[List[Dict[str, Any]]] = None,
        crossattn_cache: Optional[List[Dict[str, Any]]] = None,
        current_start: Optional[int] = 0,
        #
        memory_tokens: Optional[Tensor] = None,
        #
        rolling: bool = False,
        update_cache: bool = False,
        chunk_size: int = 1,
        #
        kv_cache_da3: Optional[List[Dict[str, Any]]] = None,
        current_start_da3: Optional[int] = 0,
        #
        clean_x: Optional[Tensor] = None,
        aug_t: Optional[Tensor] = None,
    ):
        B, (f, h, w) = noisy_latents.shape[0], noisy_latents.shape[2:]
        tff = 2 * f if self.is_causal and clean_x is not None else f
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).repeat(1, f)  # (B, f)
        timesteps = timesteps[:, :, None, None].repeat(1, 1, h//2, w//2).flatten(1)  # (B, f*hh*ww); `//2`: hard-coded for patch embeddig

        # (Optional) Plucker embedding
        if self.input_plucker:
            plucker = rearrange(plucker, "b f c h w -> (b f) c h w").to(noisy_latents.dtype)
            plucker_embeds = self.plucker_embed(plucker)
            plucker_embeds = rearrange(plucker_embeds, "(b f) c h w -> b c f h w", f=f)  # (B, D, f, hh, ww)
        else:
            plucker_embeds = None

        # (Optional) Camera token for DA3
        if self.da3_input_cam:
            assert C2W is not None and fxfycxcy is not None
            W2C = inverse_c2w(C2W).to(noisy_latents.dtype)  # (B, f, 4, 4)
            intrinsics = fxfycxcy_to_intrinsics(fxfycxcy).to(noisy_latents.dtype)  # (B, f, 3, 3)
            intrinsics[:, :, 0, 0] = intrinsics[:, :, 0, 0] * (w*8 // self.da3_down_ratio)
            intrinsics[:, :, 1, 1] = intrinsics[:, :, 1, 1] * (h*8 // self.da3_down_ratio)
            intrinsics[:, :, 0, 2] = intrinsics[:, :, 0, 2] * (w*8 // self.da3_down_ratio)
            intrinsics[:, :, 1, 2] = intrinsics[:, :, 1, 2] * (h*8 // self.da3_down_ratio)
            camera_token = self.da3_model.cam_enc(W2C, intrinsics, (h*8 // self.da3_down_ratio, w*8 // self.da3_down_ratio))  # (B, f, D)
            if self.is_causal and clean_x is not None:
                camera_token = torch.cat([camera_token, camera_token], dim=1)  # (B, 2f, D)
        else:
            camera_token = None

        # (Optional) Extra condition embedding
        if self.extra_condition_dim > 0:
            extra_condition = rearrange(extra_condition, "b f c h w -> (b f) c h w").to(noisy_latents.dtype)
            extra_condition_embeds = self.extra_condition_embed(extra_condition)
            extra_condition_embeds = rearrange(extra_condition_embeds, "(b f) c h w -> b c f h w", f=f)  # (B, D, f, hh, ww)
            if plucker is not None:
                plucker_embeds = plucker_embeds + extra_condition_embeds
            else:
                plucker_embeds = extra_condition_embeds
        else:
            extra_condition_embeds = None

        if plucker_embeds is not None:
            plucker_embeds = [plucker_embed for plucker_embed in plucker_embeds]

        # param
        device = self.model.patch_embedding.weight.device
        if self.model.freqs.device != device:
            self.model.freqs = self.model.freqs.to(device)

        # embeddings
        x = [self.model.patch_embedding(u.unsqueeze(0)) for u in noisy_latents]
        if plucker_embeds is not None:
            x = [u + v for u, v in zip(x, plucker_embeds)]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.max_seq_len is None:
            seq_len = seq_lens.max()
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if timesteps.dim() == 1:
            t = timesteps.expand(timesteps.size(0), seq_len)
        else:
            t = timesteps
        bt = t.size(0)
        e = self.model.time_embedding(
            sinusoidal_embedding_1d(self.model.freq_dim, t.flatten()).type_as(x))
        e0 = self.model.time_projection(e).unflatten(1, (6, self.model.dim)).unflatten(0, (bt, seq_len))

        # memory tokens
        if memory_tokens is not None:
            num_memory_tokens = memory_tokens.shape[1]
            t_mem = torch.zeros_like(t[:, :num_memory_tokens])
            e_mem = self.model.time_embedding(
                sinusoidal_embedding_1d(self.model.freq_dim, t_mem.flatten()).type_as(x))
            e0_mem = self.model.time_projection(e_mem).unflatten(1, (6, self.model.dim)).unflatten(0, (bt, num_memory_tokens))
            e0 = torch.cat([e0, e0_mem], dim=1)

        # context
        context_lens = None
        context = self.model.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.model.text_len - u.size(0), u.size(1))])
                for u in prompt_embeds
            ]))

        if self.is_causal:
            # Clean inputs for teacher forcing
            if clean_x is not None:
                clean_x = [self.model.patch_embedding(u.unsqueeze(0)) for u in clean_x]
                if plucker_embeds is not None:
                    clean_x = [u + v for u, v in zip(clean_x, plucker_embeds)]
                clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
                seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
                clean_x = torch.cat([
                    torch.cat([u, u.new_zeros(1, seq_lens_clean.max() - u.size(1), u.size(2))],
                            dim=1) for u in clean_x
                ])

                x = torch.cat([clean_x, x], dim=1)

                if aug_t is None:
                    aug_t = torch.zeros_like(t)
                e_clean = self.model.time_embedding(
                    sinusoidal_embedding_1d(self.model.freq_dim, aug_t.flatten()).type_as(x))
                e0_clean = self.model.time_projection(e_clean).unflatten(1, (6, self.model.dim)).unflatten(0, (bt, seq_lens_clean.max()))
                e0 = torch.cat([e0_clean, e0], dim=1)

            # Construct blockwise causal attn mask
                ## For Wan DiT
            if self.model.block_mask is None and kv_cache is None:
                if clean_x is not None:
                    self.model.block_mask = self.model._prepare_teacher_forcing_mask(
                        device,
                        num_frames=f,
                        frame_seqlen=h * w // (self.model.patch_size[1] * self.model.patch_size[2]),
                        sink_size=self.model.sink_size,
                        chunk_size=self.model.chunk_size,
                        max_attention_size=self.model.max_attention_size,
                    )
                else:
                    self.model.block_mask = self.model._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=f,
                        frame_seqlen=h * w // (self.model.patch_size[1] * self.model.patch_size[2]),
                        sink_size=self.model.sink_size,
                        chunk_size=self.model.chunk_size,
                        max_attention_size=self.model.max_attention_size,
                    )
                ## For DA3
            if self.block_mask is None and kv_cache_da3 is None:
                if clean_x is not None:
                    self.block_mask = self.model._prepare_teacher_forcing_mask(
                        device,
                        num_frames=f,
                        frame_seqlen=1 + h * w // (self.model.patch_size[1] * self.model.patch_size[2]) // (self.da3_down_ratio * self.da3_down_ratio),  # `1+` for camera token
                        sink_size=self.model.sink_size,
                        chunk_size=self.model.chunk_size,
                        max_attention_size=self.da3_max_attention_size,
                    )
                else:
                    self.block_mask = self.model._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=f,
                        frame_seqlen=1 + h * w // (self.model.patch_size[1] * self.model.patch_size[2]) // (self.da3_down_ratio * self.da3_down_ratio),  # `1+` for camera token
                        sink_size=self.model.sink_size,
                        chunk_size=self.model.chunk_size,
                        max_attention_size=self.da3_max_attention_size,
                    )

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.model.freqs,
            context=context,
            context_lens=context_lens,
        )
        if self.is_causal:
            kwargs.update(
                block_mask=self.model.block_mask,  # None if kv cache is used
            )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        dit_x, da3_x = None, None
        da3_output = []
        blocks_to_take = [11, 15, 19, 23]  # hard-coded for DA3-large
        pos, pos_nodiff = self.da3_model.backbone.pretrained._prepare_rope(B, tff, h*8//self.da3_down_ratio, w*8//self.da3_down_ratio, device)  # `8`: hard-coded for Wan2.1

        for i, block in enumerate(self.model.blocks):
            ## Only Wan DiT
            if i < len(self.model.blocks) - 24:  #  `24`: hard-coded for da3-large
                if torch.is_grad_enabled() and self.model.use_gradient_checkpointing_offload:
                    if self.is_causal and kv_cache is not None:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[i],
                                "current_start": current_start,
                                "memory_tokens": memory_tokens,
                                "rolling": rolling, "update_cache": update_cache, "chunk_size": chunk_size,
                            }
                        )
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            **kwargs,
                            use_reentrant=False,
                        )
                elif torch.is_grad_enabled() and self.model.use_gradient_checkpointing:
                    if self.is_causal and kv_cache is not None:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[i],
                                "current_start": current_start,
                                "memory_tokens": memory_tokens,
                                "rolling": rolling, "update_cache": update_cache, "chunk_size": chunk_size,
                            }
                        )
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        **kwargs,
                        use_reentrant=False,
                    )
                else:
                    if self.is_causal and kv_cache is not None:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[i],
                                "crossattn_cache": crossattn_cache[i],
                                "current_start": current_start,
                                "memory_tokens": memory_tokens,
                                "rolling": rolling, "update_cache": update_cache, "chunk_size": chunk_size,
                            }
                        )
                    x = block(x, **kwargs)
                if memory_tokens is not None:
                    x, memory_tokens = x

            ## Wan DiT & DA3
            else:
                ### Wan DiT
                dit_x = dit_x if dit_x is not None else x
                if torch.is_grad_enabled() and self.model.use_gradient_checkpointing_offload:
                    if self.is_causal and kv_cache is not None:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[i],
                                "current_start": current_start,
                                "memory_tokens": memory_tokens,
                                "rolling": rolling, "update_cache": update_cache, "chunk_size": chunk_size,
                            }
                        )
                    with torch.autograd.graph.save_on_cpu():
                        dit_x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            dit_x,
                            **kwargs,
                            use_reentrant=False,
                        )
                elif torch.is_grad_enabled() and self.model.use_gradient_checkpointing:
                    if self.is_causal and kv_cache is not None:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[i],
                                "current_start": current_start,
                                "memory_tokens": memory_tokens,
                                "rolling": rolling, "update_cache": update_cache, "chunk_size": chunk_size,
                            }
                        )
                    dit_x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        dit_x,
                        **kwargs,
                        use_reentrant=False,
                    )
                else:
                    if self.is_causal and kv_cache is not None:
                        kwargs.update(
                            {
                                "kv_cache": kv_cache[i],
                                "crossattn_cache": crossattn_cache[i],
                                "current_start": current_start,
                                "memory_tokens": memory_tokens,
                                "rolling": rolling, "update_cache": update_cache, "chunk_size": chunk_size,
                            }
                        )
                    dit_x = block(dit_x, **kwargs)
                if memory_tokens is not None:
                    dit_x, memory_tokens = dit_x

                ### DA3
                if da3_x is None:  # optional: downsample DiT features for DA3 input
                    da3_x = rearrange(x, "b (f h w) d -> (b f) d h w", f=tff, h=h//2, w=w//2)  # `2`: hard-coded for patch embedding
                    if self.da3_down_ratio > 1:
                        da3_x = tF.interpolate(da3_x, scale_factor=1/self.da3_down_ratio, mode="bilinear", align_corners=True)
                    da3_x = rearrange(da3_x, "(b f) d h w -> b (f h w) d", f=tff, h=h//(2*self.da3_down_ratio), w=w//(2*self.da3_down_ratio))  # `2`: hard-coded for patch embedding
                else:
                    da3_x = da3_x
                da3_i = i - (len(self.model.blocks) - 24)
                da3_block = self.da3_model.backbone.pretrained.blocks[da3_i]

                if da3_i == 0:  # the first layer of DA3
                    B = da3_x.shape[0]
                    da3_x = rearrange(da3_x, "b (f h w) d -> (b f) (h w) d", f=tff, h=h//(2*self.da3_down_ratio), w=w//(2*self.da3_down_ratio))  # `2`: hard-coded for patch embedding
                    da3_x = self.da3_adapter(da3_x)  # align dimension
                    cls_token = self.da3_model.backbone.pretrained.prepare_cls_token(B, tff)
                    da3_x = torch.cat((cls_token, da3_x), dim=1)
                    da3_x = da3_x + self.da3_model.backbone.pretrained.interpolate_pos_encoding(da3_x, h*8//self.da3_down_ratio, w*8//self.da3_down_ratio)  # `8`: hard-coded for Wan2.1
                    da3_x = rearrange(da3_x, "(b f) n d -> b f n d", f=tff)

                if da3_i < 8:  # `8`: hard-coded for DA3-large `self.rope_start`
                    g_pos, l_pos = None, None
                else:
                    g_pos, l_pos = pos_nodiff, pos

                if da3_i == 8:  # `8`: hard-coded for DA3-large `self.alt_start`
                    if camera_token is None:
                        ref_token = self.da3_model.backbone.pretrained.camera_token[:, :1].expand(B, -1, -1)
                        src_token = self.da3_model.backbone.pretrained.camera_token[:, 1:].expand(B, tff - 1, -1)
                        camera_token = torch.cat([ref_token, src_token], dim=1)
                    da3_x[:, :, 0, :] = camera_token

                if da3_i >= 8 and da3_i % 2 == 1:  # `8`: hard-coded for DA3-large `self.alt_start`
                    da3_x = self.da3_model.backbone.pretrained.process_attention(
                        da3_x, da3_block, "global", pos=g_pos,
                        gradient_checkpointing=self.training,
                        #
                        block_mask=self.block_mask if self.is_causal else None,
                        kv_cache=kv_cache_da3[da3_i] if self.is_causal and kv_cache_da3 is not None else None,
                        current_start=current_start_da3,
                        frame_seqlen=da3_x.shape[2],  # it should be h*w//4 + 1
                        #
                        rolling=rolling, update_cache=update_cache, chunk_size=chunk_size,
                    )
                else:
                    da3_x = self.da3_model.backbone.pretrained.process_attention(
                        da3_x, da3_block, "local", pos=l_pos,
                        gradient_checkpointing=self.training,
                        #
                        # NOTE: no need causality for local attention
                        # block_mask=self.block_mask if self.is_causal else None,
                        # kv_cache=kv_cache_da3[da3_i] if self.is_causal and kv_cache_da3 is not None else None,
                        # current_start=current_start_da3,
                        # frame_seqlen=da3_x.shape[2],  # it should be h*w//4 + 1
                        #
                        # rolling=rolling, update_cache=update_cache, chunk_size=chunk_size,
                    )
                    local_da3_x = da3_x

                if da3_i in blocks_to_take:
                    out_da3_x = torch.cat([local_da3_x, da3_x], dim=-1)
                    da3_output.append((out_da3_x[:, :, 0], out_da3_x))

                ### (Optional) Interaction
                if self.da3_interactive:
                    dit_x = rearrange(dit_x, "b (f h w) d -> (b f) (h w) d", f=tff, h=h//2, w=w//2)  # `2`: hard-coded for patch embedding
                    da3_x = rearrange(da3_x, "b f n d -> (b f) n d")
                    dit_x_res, da3_x_res = self.dit_da3_interactive[da3_i](dit_x, da3_x[:, 1:, :], h=h//2, w=w//2)
                    dit_x, da3_x = dit_x + dit_x_res, torch.cat([da3_x[:, :1, :], da3_x[:, 1:, :] + da3_x_res], dim=1)
                    dit_x = rearrange(dit_x, "(b f) (h w) d -> b (f h w) d", f=tff, h=h//2, w=w//2)  # `2`: hard-coded for patch embedding
                    da3_x = rearrange(da3_x, "(b f) n d -> b f n d", f=tff)

        # Wan DiT head & unpatchify
        if self.is_causal and clean_x is not None:
            dit_x = dit_x[:, dit_x.shape[1] // 2:, :]  # remove teacher forcing part
        dit_x = self.model.head(dit_x, e.unflatten(0, (bt, seq_len)))
        dit_x = self.model.unpatchify(dit_x, grid_sizes)
        dit_x = torch.stack([u.float() for u in dit_x])

        # DA3 head & unpatchify
        camera_tokens = [out[0] for out in da3_output]
        da3_outputs = [
            torch.cat(
                [
                    # `1024`: hard-coded for DA3-large
                    out[1][..., :1024],
                    self.da3_model.backbone.pretrained.norm(out[1][..., 1024:])
                ],
                dim=-1,
            )
            for out in da3_output
        ]
        da3_outputs = [out[..., 1:, :] for out in da3_outputs]
        # NOTE: keep teacher forcing part for DA3
        # da3_outputs = [out[:, out.shape[1]//2:, :] for out in da3_outputs]  # remove teacher forcing part
        feats = tuple(zip(da3_outputs, camera_tokens))
        head_outputs = self.da3_model.head(feats, h*8//self.da3_down_ratio, w*8//self.da3_down_ratio, patch_start_idx=0, chunk_size=self.da3_chunk_size)  # `*8`: hard-coded for Wan2.1
        depths, depths_conf = head_outputs["depth"], head_outputs["depth_conf"]  # (B, f, H, W), (B, f, H, W)
        rays, rays_conf = head_outputs["ray"], head_outputs["ray_conf"]  # (B, f, H, W, 6), (B, f, H, W)

        # Camera
        pose_enc = self.da3_model.cam_dec(feats[-1][1])  # (B, f, 9)
        with torch.no_grad():
            ## Camera decoder
            if not self.da3_use_ray_pose:
                R, T = quat_to_mat(pose_enc[..., 3:7]), pose_enc[..., :3]
                C2W = torch.cat([R, T[..., None]], dim=-1)  # (B, f, 3, 4)
                C2W = torch.cat([C2W, torch.zeros_like(C2W[..., :1, :])], dim=-2)  # (B, f, 4, 4)
                C2W[..., 3, 3] = 1.  # (B, f, 4, 4)

                fov_h, fov_w = pose_enc[..., 7], pose_enc[..., 8]
                fx, fy = 0.5 / torch.clamp(torch.tan(fov_w / 2.), 1e-6), 0.5 / torch.clamp(torch.tan(fov_h / 2.), 1e-6)
                cx, cy = 0.5 * torch.ones_like(fov_h), 0.5 * torch.ones_like(fov_w)
                fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1)  # (B, f, 4)
            ## Raymap
            else:
                pred_extrinsic, pred_focal_lengths, pred_principal_points = \
                    get_extrinsic_from_camray(rays, rays_conf, rays.shape[-3], rays.shape[-2])

                C2W = pred_extrinsic  # (B, f, 4, 4)
                fxfycxcy = torch.stack([
                    pred_focal_lengths[:, :, 0] / 2,
                    pred_focal_lengths[:, :, 1] / 2,
                    pred_principal_points[:, :, 0] / 2,
                    pred_principal_points[:, :, 1] / 2,
                ], dim=-1)  # (B, f, 4)

        rays = rearrange(rays, "b f h w c -> b f c h w")  # (B, f, 6, H/2, W/2)
        da3_final_outputs = {
            "depth": depths,            # (B, f, H, W)
            "depth_conf": depths_conf,  # (B, f, H, W)
            "ray": rays,                # (B, f, 6, H/2, W/2)
            "ray_conf": rays_conf,      # (B, f, H/2, W/2)
            "pose_enc": pose_enc,       # (B, f, 9)
            #
            "C2W": C2W,                 # (B, f, 4, 4)
            "fxfycxcy": fxfycxcy,       # (B, f, 4)
        }

        if memory_tokens is not None:
            return (dit_x, da3_final_outputs), memory_tokens
        else:
            return dit_x, da3_final_outputs

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
        sigma_t = sigmas[timestep_id]

        # NOTE: handle (B*f,) input shape and timestep=0 for I2V
        sigma_t[timestep == 0.] = 0.
        sigma_t = sigma_t.reshape(-1, 1, 1, 1)

        x0_pred = xt - sigma_t * flow_pred
        x0_pred = rearrange(x0_pred, "(b f) d h w -> b d f h w", f=f)
        return x0_pred.to(original_dtype)


class InteractiveModule(nn.Module):
    def __init__(self, dim1: int, dim2: int, dim: int, num_heads: int, down_ratio=1, qk_norm=True, eps=1e-6):
        super().__init__()

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.down_ratio = down_ratio
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        # self.q1, self.q2 = nn.Linear(dim1, dim), nn.Linear(dim2, dim)
        # self.k1, self.k2 = nn.Linear(dim1, dim), nn.Linear(dim2, dim)
        # self.v1, self.v2 = nn.Linear(dim1, dim), nn.Linear(dim2, dim)
        # self.o1, self.o2 = nn.Linear(dim, dim1), nn.Linear(dim, dim2)
        # if qk_norm:
        #     self.norm_q1, self.norm_q2 = WanRMSNorm(self.head_dim, eps), WanRMSNorm(self.head_dim, eps)
        #     self.norm_k1, self.norm_k2 = WanRMSNorm(self.head_dim, eps), WanRMSNorm(self.head_dim, eps)
        # else:
        #     self.norm_q1, self.norm_q2 = nn.Identity(), nn.Identity()
        #     self.norm_k1, self.norm_k2 = nn.Identity(), nn.Identity()
        self.o1 = nn.Sequential(
            nn.Linear(dim1+dim2, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim1),
        )
        self.o2 = nn.Sequential(
            nn.Linear(dim1+dim2, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim2),
        )

        zero_init_module(self.o1)
        zero_init_module(self.o2)

    def forward(self, x1: Tensor, x2: Tensor, h=None, w=None) -> Tuple[Tensor, Tensor]:
        # reshape_fn = lambda x: rearrange(x, "b n (h hd) -> b n h hd", h=self.num_heads)

        # q1, k1, v1 = reshape_fn(self.q1(x1)), reshape_fn(self.k1(x1)), reshape_fn(self.v1(x1))
        # q1, k1 = self.norm_q1(q1), self.norm_k1(k1)

        # q2, k2, v2 = reshape_fn(self.q2(x2)), reshape_fn(self.k2(x2)), reshape_fn(self.v2(x2))
        # q2, k2 = self.norm_q2(q2), self.norm_k2(k2)

        # o1 = self.o1(attention(q1, k2, v2).flatten(2))
        # o2 = self.o2(attention(q2, k1, v1).flatten(2))

        if self.down_ratio > 1:
            assert h is not None and w is not None
            x2 = rearrange(x2, "b (h w) d -> b d h w", h=h//self.down_ratio, w=w//self.down_ratio)
            x2 = tF.interpolate(x2, scale_factor=self.down_ratio, mode="bilinear", align_corners=True)
            x2 = rearrange(x2, "b d h w -> b (h w) d")

        o1 = self.o1(torch.cat([x1, x2], dim=-1))
        o2 = self.o2(torch.cat([x1, x2], dim=-1))

        if self.down_ratio > 1:
            o2 = rearrange(o2, "b (h w) d -> b d h w", h=h, w=w)
            o2 = tF.interpolate(o2, scale_factor=1/self.down_ratio, mode="bilinear", align_corners=True)
            o2 = rearrange(o2, "b d h w -> b (h w) d")

        return o1, o2
