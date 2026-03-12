from typing import *
from torch import Tensor

import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as tF
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from einops import rearrange
from pytorch_msssim import ssim as SSIM
from lpips import LPIPS

from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options, ROOT
from src.models.modules import (
    WanTextEncoderWrapper,
    WanVAEWrapper,
    WanDiffusionWrapper,
    WanDiffusionDA3Wrapper,
    VAEDecoderWrapper,
    TAEHV,
)
from src.models.modules.decoder_wrapper import ZERO_VAE_CACHE_512, ZERO_VAE_CACHE
from src.models.losses import XYZLoss, DepthLoss, CameraLoss
from src.utils.ema import EMAParams
from src.utils import convert_to_buffer, plucker_ray, colorize_depth, filter_da3_points, render_pt3d_points, mv_interpolate
from src.utils.distributed import get_sp_world_size, barrier


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Wan(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        if opt.use_vidprom:
            assert not opt.first_latent_cond
            with open(f"{ROOT}/.cache/vidprom_filtered_extended.txt", encoding="utf-8") as f:
                self.prompt_list = [line.rstrip() for line in f]
        else:
            self.prompt_list = None

        # (Optinoal) Causal VAE decoder
        if opt.is_causal and opt.input_pcrender and opt.load_da3:
            if not opt.load_tae:
                self.current_vae_decoder = VAEDecoderWrapper()
                vae_state_dict = torch.load(opt.vae_path, map_location="cpu", weights_only=True)
                decoder_state_dict = {}
                for key, value in vae_state_dict.items():
                    if "decoder." in key or "conv2" in key:
                        decoder_state_dict[key] = value
                self.current_vae_decoder.load_state_dict(decoder_state_dict)
            else:
                class DotDict(dict):
                    __getattr__ = dict.__getitem__
                    __setattr__ = dict.__setitem__
                class TAEHVDiffusersWrapper(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.taehv = TAEHV(checkpoint_path=opt.tae_path)
                        self.config = DotDict(scaling_factor=1.)
                    def decode(self, latents: Tensor):
                        # n, c, t, h, w = latents.shape
                        # low-memory, set parallel=True for faster + higher memory
                        latents = rearrange(latents, "b c f h w -> b f c h w")
                        return self.taehv.decode_video(latents, parallel=False).clamp(0., 1.)
                self.current_vae_decoder = TAEHVDiffusersWrapper()

            self.current_vae_decoder.requires_grad_(False)
            self.current_vae_decoder.eval()
        else:
            self.current_vae_decoder = None

        # Text encoder
        if opt.load_text_encoder:
            self.text_encoder = WanTextEncoderWrapper(opt.wan_dir)
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        else:
            self.text_encoder = None

        # Diffusion model
        if not opt.load_da3:
            self.diffusion = WanDiffusionWrapper(
                opt.wan_dir,
                opt.num_train_timesteps,
                opt.num_inference_steps,
                opt.shift,
                0.,    # hard-coded `sigma_min`
                True,  # hard-coded `extra_one_step`
                #
                opt.input_plucker,
                opt.extra_condition_dim,
                #
                opt.use_gradient_checkpointing,
                opt.use_gradient_checkpointing_offload,
                #
                is_causal=opt.is_causal,
                sink_size=opt.sink_size,
                chunk_size=opt.chunk_size,
                max_attention_size=opt.max_attention_size,
                rope_outside=opt.rope_outside,
                #
                feat_proj=opt.self_supervised_loss_weight > 0.,
                use_ddt=opt.use_ddt,
                ddt_num_layers=opt.ddt_num_layers,
                ddt_fusion=opt.ddt_fusion,
            )
        else:
            self.diffusion = WanDiffusionDA3Wrapper(
                opt.wan_dir,
                opt.num_train_timesteps,
                opt.num_inference_steps,
                opt.shift,
                0.,    # hard-coded `sigma_min`
                True,  # hard-coded `extra_one_step`
                #
                opt.input_plucker,
                opt.extra_condition_dim,
                #
                opt.use_gradient_checkpointing,
                opt.use_gradient_checkpointing_offload,
                #
                is_causal=opt.is_causal,
                sink_size=opt.sink_size,
                chunk_size=opt.chunk_size,
                max_attention_size=opt.max_attention_size,
                rope_outside=opt.rope_outside,
                #
                da3_model_name=opt.da3_model_name,
                da3_chunk_size=opt.da3_chunk_size,
                da3_down_ratio=opt.da3_down_ratio,
                da3_use_ray_pose=opt.da3_use_ray_pose,
                da3_interactive=opt.da3_interactive and not opt.only_train_da3,
                da3_input_cam=opt.da3_input_cam,
                da3_max_attention_size=opt.da3_max_attention_size,
            )
            ## Freeze DA3 camera encoder / token
            if self.diffusion.da3_model.cam_enc is not None:
                self.diffusion.da3_model.cam_enc.requires_grad_(False)
            if self.diffusion.da3_model.backbone.pretrained.camera_token is not None:
                self.diffusion.da3_model.backbone.pretrained.camera_token.requires_grad_(False)

            ## (Optional) Freeze some layers of DiT or DA3
            if opt.only_train_da3:
                self.diffusion.requires_grad_(False)
                self.diffusion.da3_adapter.requires_grad_(True)
                self.diffusion.da3_model.requires_grad_(True)
            if opt.fix_da3_heads:
                self.diffusion.da3_model.head.requires_grad_(False)
                self.diffusion.da3_model.cam_dec.requires_grad_(False)

            ## DA3 losses
            self.ray_loss_fn, self.depth_loss_fn, self.camera_loss_fn = \
                XYZLoss(opt), DepthLoss(opt), CameraLoss(opt)

        # (Optional) Freeze some layers of DiT
        if opt.only_train_resdit:
            self.diffusion.requires_grad_(False)
            for i, block in enumerate(self.diffusion.model.blocks):
                if i >= len(self.diffusion.model.blocks) - 24:  #  `24`: hard-coded for da3-large
                    block.requires_grad_(True)
        if opt.fix_shared_dit_layers:
            for i, block in enumerate(self.diffusion.model.blocks):
                if i < len(self.diffusion.model.blocks) - 24:  #  `24`: hard-coded for da3-large
                    block.requires_grad_(False)

        if opt.generator_path is not None:
            state_dict = torch.load(opt.generator_path, map_location="cpu", weights_only=True)
            if "generator_ema" in state_dict:
                self.diffusion.load_state_dict(state_dict["generator_ema"], strict=False)
            elif "generator" in state_dict:
                self.diffusion.load_state_dict(state_dict["generator"], strict=False)
            else:
                self.diffusion.load_state_dict(state_dict, strict=False)

        # Add LoRA in the diffusion model, will freeze all parameters except LoRA layers
        if opt.use_lora_in_wan:
            self._add_lora_to_wan(
                target_modules=opt.lora_target_modules_in_wan.split(","),
                lora_rank=opt.lora_rank_in_wan,
            )
            # Load LoRA checkpoint if specified
            if opt.lora_path is not None:
                lora_state_dict = torch.load(opt.lora_path, map_location="cpu", weights_only=True)
                self.load_lora_weights(lora_state_dict, strict=True)

        # Set other trainable parameters except LoRA layers in the diffusion model
        if opt.more_trainable_wan_params is not None:
            trainble_names = opt.more_trainable_wan_params.split(",")
            if opt.use_lora_in_wan:
                trainble_names.append("lora")
            for name, param in self.diffusion.named_parameters():
                _flag = False
                for trainble_name in trainble_names:
                    if trainble_name in name:
                        param.requires_grad_(True)
                        _flag = True
                        break
                if not _flag:
                    param.requires_grad_(False)

        # LPIPS for evaluation
        if self.opt.use_lpips:
            self.lpips_loss = LPIPS(net="vgg")
            convert_to_buffer(self.lpips_loss, persistent=False)  # no gradient & not save to checkpoint
        else:
            self.lpips_loss = None

        # (Optional) RIFLEx (https://arxiv.org/pdf/2502.15894) for length extrapolation
        if opt.enable_riflex:
            self.diffusion.model.enable_riflex()

        self.kv_cache_pos, self.kv_cache_neg = None, None
        self.crossattn_cache_pos, self.crossattn_cache_neg = None, None
        self.kv_cache_pos_da3, self.kv_cache_neg_da3 = None, None

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self,
        data: Dict[str, Any],
        dtype: torch.dtype = torch.float32,
        is_eval: bool = False,
        vae: Optional[WanVAEWrapper] = None,
        ema_params: Optional[EMAParams] = None,
    ):
        outputs = {}
        device = self.diffusion.model.device

        # For multi-clip generation
        data, clip_latent_lens = self._multiclip_batch(data)
        # Determine actual num_clips from data
        actual_num_clips = len(data["prompt"][0]) if isinstance(data["prompt"][0], list) else 1
        if actual_num_clips > 1:
            assert clip_latent_lens is not None

        if "image" in data:
            images = data["image"].to(device=device, dtype=dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W) = images.shape
        else:
            B = len(data["prompt"])
            F, H, W = (self.opt.num_input_frames - 1) * actual_num_clips + 1, self.opt.input_res[0], self.opt.input_res[1]

        idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
        if self.opt.load_da3:
            assert "depth" in data
            gt_depths = data["depth"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
        if "C2W" in data and "fxfycxcy" in data:
            C2W = data["C2W"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 4, 4)
            fxfycxcy = data["fxfycxcy"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 4)
            plucker = plucker_ray(H, W, C2W.float(), fxfycxcy.float())[0].to(dtype)  # (B, f, 6, H, W)
        else:
            C2W, fxfycxcy, plucker = None, None, None

        # Text encoder
        if self.text_encoder is not None:
            prompts = data["prompt"]  # a list of strings
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                self.text_encoder.eval()
                prompt_embeds = self._encode_prompt_batch(prompts)  # (B, N=512, D') or (B, num_clips, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if "image" in data:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                latents = self.encode(images * 2. - 1., vae)  # (B, D, f, h, w)
                if self.opt.first_latent_cond:
                    cond_latents = latents[:, :, 0:1, :, :].clone()  # (B, D, 1, h, w)
                else:
                    cond_latents = None
        else:
            latents = torch.zeros(
                B,
                self.opt.latent_dim,
                1 + (F - 1) // self.opt.compression_ratio[0],
                H // self.opt.compression_ratio[1],
                W // self.opt.compression_ratio[2],
            ).to(device=device, dtype=dtype)  # to provide shape info only
            cond_latents = None
        f = latents.shape[2]
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Point cloud rendering
        if self.opt.input_pcrender:
            assert "depth" in data and "conf" in data and C2W is not None and fxfycxcy is not None
            depths = data["depth"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
            confs = data["conf"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
            images_f = images[:, idxs, ...]  # (B, f, 3, H, W)
            if self.opt.da3_down_ratio != 1:
                images_f = mv_interpolate(images_f,
                    size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)  # (B, f, 3, H, W)
            with torch.no_grad():
                ## Bidirectional rendering
                if not self.opt.is_causal:
                    all_render_images, all_render_depths = [], []
                    for i in range(B):
                        points, colors = filter_da3_points(
                            images_f[i], depths[i], confs[i], C2W[i], fxfycxcy[i],
                            conf_thresh_percentile=self.opt.conf_thresh_percentile,
                            random_sample_ratio=self.opt.rand_pcrender_ratio,
                            min_num_points=self.opt.min_num_points,
                            max_num_points=self.opt.max_num_points,
                        )
                        render_images, render_depths = render_pt3d_points(
                            H, W, points, colors, C2W[i], fxfycxcy[i],
                            return_depth=True,
                        )  # (f, 3, H, W) in [0, 1]; (f, H, W)
                        all_render_images.append(render_images.to(dtype))
                        all_render_depths.append(render_depths.to(dtype))
                    render_images = torch.stack(all_render_images, dim=0)  # (B, f, 3, H, W) in [0, 1]
                    render_depths = torch.stack(all_render_depths, dim=0)  # (B, f, H, W)
                ## Teacher forcing style rendering
                else:
                    all_render_images, all_render_depths = [], []
                    for i in range(B):
                        assert f % self.opt.chunk_size == 0
                        num_chunks = f // self.opt.chunk_size
                        all_render_images_chunk, all_render_depths_chunk, all_points, all_colors = [], [], None, None
                        for ci in range(num_chunks-1):
                            chunk_idxs = torch.arange(ci * self.opt.chunk_size, (ci + 1) * self.opt.chunk_size).to(device=device, dtype=torch.long)
                            next_chunk_idxs = torch.arange((ci + 1) * self.opt.chunk_size, (ci + 2) * self.opt.chunk_size).to(device=device, dtype=torch.long)
                            points, colors = filter_da3_points(
                                images_f[i, chunk_idxs], depths[i, chunk_idxs], confs[i, chunk_idxs], C2W[i, chunk_idxs], fxfycxcy[i, chunk_idxs],
                                conf_thresh_percentile=self.opt.conf_thresh_percentile,
                                random_sample_ratio=self.opt.rand_pcrender_ratio,
                                min_num_points=self.opt.min_num_points,
                                max_num_points=self.opt.max_num_points,
                            )
                            if all_points is None:
                                all_points, all_colors = points, colors
                            else:
                                all_points = torch.cat([all_points, points], dim=0)
                                all_colors = torch.cat([all_colors, colors], dim=0)
                            render_images, render_depths = render_pt3d_points(
                                H, W, all_points, all_colors,
                                C2W[i, next_chunk_idxs], fxfycxcy[i, next_chunk_idxs],
                                return_depth=True
                            )  # (f_chunk, 3, H, W) in [0, 1]; (f_chunk, H, W)
                            all_render_images_chunk.append(render_images.to(dtype))
                            all_render_depths_chunk.append(render_depths.to(dtype))
                        all_render_images_chunk = torch.cat(all_render_images_chunk, dim=0)  # (f-f_chunk, 3, H, W)
                        all_render_depths_chunk = torch.cat(all_render_depths_chunk, dim=0)  # (f-f_chunk, H, W)

                        ### For the first chunk
                        if cond_latents is not None:
                            points, colors = filter_da3_points(
                                images_f[i, 0:1], depths[i, 0:1], confs[i, 0:1], C2W[i, 0:1], fxfycxcy[i, 0:1],
                                conf_thresh_percentile=self.opt.conf_thresh_percentile,
                                random_sample_ratio=self.opt.rand_pcrender_ratio,
                                min_num_points=self.opt.min_num_points,
                                max_num_points=self.opt.max_num_points,
                                all_valid=True,  # save all points for image conditioning
                            )
                            first_render_images, first_render_depths = render_pt3d_points(
                                    H, W, points, colors,
                                    C2W[i, :self.opt.chunk_size], fxfycxcy[i, :self.opt.chunk_size],
                                    return_depth=True
                                )
                        else:
                            first_render_images = torch.zeros_like(all_render_images_chunk[:self.opt.chunk_size, ...])
                            first_render_depths = torch.zeros_like(all_render_depths_chunk[:self.opt.chunk_size, ...])

                        all_render_images_chunk = torch.cat([first_render_images, all_render_images_chunk], dim=0)  # (f, 3, H, W)
                        all_render_depths_chunk = torch.cat([first_render_depths, all_render_depths_chunk], dim=0)  # (f, H, W)

                        all_render_images.append(all_render_images_chunk)
                        all_render_depths.append(all_render_depths_chunk)
                    render_images = torch.stack(all_render_images, dim=0)  # (B, f, 3, H, W)
                    render_depths = torch.stack(all_render_depths, dim=0)  # (B, f, H, W)
        else:
            render_images, render_depths = None, None

        # (Optional) Extra conditioning: image + depth + mask
        if render_images is not None:
            input_extra_condition = torch.cat([
                render_images, render_depths.unsqueeze(2),
                (render_depths > 0.).unsqueeze(2).to(dtype),
            ], dim=2)  # (B, f, 3+1+1, H, W)
        else:
            input_extra_condition = None

        # Diffusion
        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)
        noises = torch.randn_like(latents)

        min_t, max_t = int(self.opt.min_timestep_boundary * self.opt.num_train_timesteps), \
            int(self.opt.max_timestep_boundary * self.opt.num_train_timesteps)
        if self.opt.self_supervised_loss_weight <= 0.:
            if not self.opt.is_causal:
                timesteps_id = torch.randint(min_t, max_t, (1,))  # (1,); batch share the same timestep for simpler time scheduler
                timesteps_id = timesteps_id.unsqueeze(1).repeat(B, f)  # (B, f)
            else:  # teacher / diffusion forcing
                assert f % self.opt.chunk_size == 0
                num_chunks = f // self.opt.chunk_size

                timesteps_id = torch.randint(min_t, max_t, (num_chunks,))  # (num_chunks,); each chunk in different noise level
                timesteps_id = timesteps_id.repeat_interleave(self.opt.chunk_size, dim=0).repeat(B, 1)  # (B, f); batch share the same timestep for simpler time scheduler
        else:
            # cf. Self-Flow: for self supervision
            assert f % self.opt.chunk_size == 0
            num_chunks = f // self.opt.chunk_size

            # Randomly select two timesteps
            timesteps_id_pair = torch.randint(min_t, max_t, (2,))  # (2,)
            timesteps_id_small = timesteps_id_pair.min()  # smaller timestep (less noise)
            timesteps_id_large = timesteps_id_pair.max()  # larger timestep (more noise)

            # Randomly select which chunks to mask with small timestep
            num_masked_chunks = int(num_chunks * self.opt.self_supervised_mask_ratio)
            mask_indices = torch.randperm(num_chunks)[:num_masked_chunks]  # randomly select chunks to mask

            # Create chunk-level timestep assignment
            chunk_timesteps = torch.full((num_chunks,), timesteps_id_large, dtype=torch.long)  # default: large timestep
            chunk_timesteps[mask_indices] = timesteps_id_small  # masked chunks: small timestep

            # Expand to frame-level: (num_chunks,) -> (f,) -> (B, f)
            timesteps_id = chunk_timesteps.repeat_interleave(self.opt.chunk_size, dim=0).repeat(B, 1)  # (B, f)
        timesteps = self.diffusion.scheduler.timesteps[timesteps_id].to(dtype=dtype, device=device)
        if self.opt.no_noise_for_da3:  # to train da3 in clean latents
            timesteps = torch.zeros_like(timesteps)
        if cond_latents is not None:
            timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

        noisy_latents = self.diffusion.scheduler.add_noise(
            latents.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
            noises.transpose(1, 2).flatten(0, 1),   # (B*f, D, h, w)
            timesteps.flatten(0, 1),                # (B*f,)
        ).detach().unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)
        targets = self.diffusion.scheduler.training_target(latents, noises)

        # # Classifier-free guidance dropout
        # if self.training:
        #     masks = (torch.rand(B, device=device) < self.opt.cfg_dropout).to(dtype)
        #     prompt_embeds = prompt_embeds * masks[:, None, None]# + negative_prompt_embeds * (1 - masks)[:, None, None]

        model_outputs = self.diffusion(
            noisy_latents,
            timesteps,
            prompt_embeds,
            plucker=plucker if self.opt.input_plucker else None,
            C2W=C2W, fxfycxcy=fxfycxcy,  # for DA3
            extra_condition=input_extra_condition,
            #
            clean_x=latents if self.opt.use_teacher_forcing else None,
            #
            clip_latent_lens=clip_latent_lens,  # for multi-clip generation
            #
            return_feat_layer_idx=self.opt.student_layer_idx,  # for self-supervise
        )

        if self.opt.student_layer_idx is not None:
            model_outputs, student_feats = model_outputs

        if self.opt.load_da3:
            model_outputs, da3_outputs = model_outputs

        # Diffusion loss
        diffusion_loss = tF.mse_loss(model_outputs.float(), targets.float(), reduction="none")  # (B, D, f, h, w)
        diffusion_loss = self.diffusion.scheduler.training_weight(timesteps.flatten(0, 1)).reshape(-1, 1, 1, 1) * \
            diffusion_loss.transpose(1, 2).flatten(0, 1)  # (B*f, D, h, w)
        diffusion_loss = diffusion_loss.unflatten(0, (B, f)).transpose(1, 2)  # (B, D, f, h, w)
        outputs["diffusion_loss"] = diffusion_loss.mean()
        outputs["loss"] = outputs["diffusion_loss"]

        # Self-supervised loss
        if self.opt.self_supervised_loss_weight > 0.:
            assert ema_params is not None
            # Store the model parameters temporarily and load the EMA parameters
            ema_params.cache_model(cpu=False)
            ema_params.copy_to_model()
            barrier()  # make sure all processes have finished the above operations before evaluation

            # Asymmetric noise for teacher inputs
            teacher_timesteps_id = timesteps_id_small.unsqueeze(0).unsqueeze(1).repeat(B, f)  # (B, f)
            teacher_timesteps = self.diffusion.scheduler.timesteps[teacher_timesteps_id].to(dtype=dtype, device=device)
            if cond_latents is not None:
                teacher_timesteps = torch.cat([torch.zeros_like(teacher_timesteps[:, :1]), teacher_timesteps[:, 1:]], dim=1)

            teacher_noisy_latents = self.diffusion.scheduler.add_noise(
                latents.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
                noises.transpose(1, 2).flatten(0, 1),   # (B*f, D, h, w)
                teacher_timesteps.flatten(0, 1),        # (B*f,)
            ).detach().unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)

            with torch.no_grad():
                model_outputs, teacher_feats = self.diffusion(
                    teacher_noisy_latents,
                    teacher_timesteps,
                    prompt_embeds,
                    plucker=plucker if self.opt.input_plucker else None,
                    C2W=C2W, fxfycxcy=fxfycxcy,  # for DA3
                    extra_condition=input_extra_condition,
                    #
                    clean_x=latents if self.opt.use_teacher_forcing else None,
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    #
                    return_feat_layer_idx=self.opt.teacher_layer_idx,  # for self-supervise
                )
            outputs["ss_loss"] = -tF.cosine_similarity(student_feats.float(), teacher_feats.float(), dim=-1).mean()
            outputs["loss"] = outputs["diffusion_loss"] + self.opt.self_supervised_loss_weight * outputs["ss_loss"]

            # Switch back to the original model parameters
            ema_params.restore_model_from_cache()
            barrier()  # make sure all processes have finished restoring the model parameters before the next training step

        # (Optional) DA3 loss
        if self.opt.load_da3:
            ## Get ground-truth geometry labels
            _, (ray_o, ray_d) = plucker_ray(H//2//self.opt.da3_down_ratio, W//2//self.opt.da3_down_ratio,
                C2W.float(), fxfycxcy.float(), normalize_ray_d=False)
            gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
            gt_pose_enc = torch.cat([
                C2W[:, :, :3, 3].float(),  # (B, f, 3)
                mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
                2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
                2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
            ], dim=-1).to(dtype)  # (B, f, 9)
            if self.opt.use_teacher_forcing:
                gt_depths, gt_raymaps, gt_pose_enc = \
                    torch.cat([gt_depths] * 2, dim=1), torch.cat([gt_raymaps] * 2, dim=1), torch.cat([gt_pose_enc] * 2, dim=1)
            ## Compute geometry losses
            depth_loss = self.depth_loss_fn(da3_outputs["depth"], gt_depths, confs=da3_outputs["depth_conf"])  # (B, f)
            ray_loss = self.ray_loss_fn(da3_outputs["ray"], gt_raymaps, confs=da3_outputs["ray_conf"])  # (B, f)
            camera_loss = self.camera_loss_fn(da3_outputs["pose_enc"], gt_pose_enc)  # (B, f)
            if self.opt.no_noise_for_da3:
                da3_loss = (depth_loss + ray_loss + camera_loss).flatten(0, 1)  # (B*f,)
            else:  # weighted by noise level
                if self.opt.da3_weight_type == "uniform":
                    da3_weights = 1.
                elif self.opt.da3_weight_type == "diffusion":
                    da3_weights = self.diffusion.scheduler.training_weight(timesteps.flatten(0, 1))
                elif self.opt.da3_weight_type == "inverse_timestep":
                    da3_weights = 1. / (timesteps.flatten(0, 1) + 0.1)
                da3_loss = da3_weights * (depth_loss + ray_loss + camera_loss).flatten(0, 1)  # (B*f,)
            outputs["depth_loss"] = depth_loss.mean()
            outputs["ray_loss"] = ray_loss.mean()
            outputs["camera_loss"] = camera_loss.mean()
            outputs["loss"] = outputs["diffusion_loss"] + da3_loss.mean()

        # # For visualizaiton
        # if is_eval:
        #     pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_latents, timesteps).to(dtype)
        #     outputs["images_predx0"] = (self.decode_latent(pred_x0, vae).clamp(-1., 1.) + 1.) / 2.
        #     if "image" in data:
        #         outputs["images_recon"] = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.

        #     if self.opt.load_da3:
        #         outputs["images_gt_depth"] = colorize_depth(1./gt_depths, batch_mode=True)
        #         outputs["images_pred_depth"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)

        #     if render_images is not None:
        #         outputs["images_render"] = render_images

        return outputs

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def evaluate(self, data: Dict[str, Any], dtype: torch.dtype = torch.bfloat16, verbose: bool = True, vae: Optional[WanVAEWrapper] = None):
        verbose = verbose and (not dist.is_initialized() or dist.get_rank() == 0)
        if self.opt.is_causal:
            return self.evaluate_causal(data, dtype, verbose, vae)
        else:
            return self.evaluate_bidirectional(data, dtype, verbose, vae)

    def evaluate_bidirectional(self, data: Dict[str, Any], dtype: torch.dtype, verbose: bool = True, vae: Optional[WanVAEWrapper] = None):
        outputs = {}
        device = self.diffusion.model.device

        # For multi-clip generation
        data, clip_latent_lens = self._multiclip_batch(data)
        # Determine actual num_clips from data
        actual_num_clips = len(data["prompt"][0]) if isinstance(data["prompt"][0], list) else 1
        if actual_num_clips > 1:
            assert clip_latent_lens is not None

        self.diffusion.eval()
        self.diffusion.scheduler.set_timesteps(self.opt.num_inference_steps, training=False)

        if "image" in data:
            images = data["image"].to(device=device, dtype=dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W) = images.shape

            # For visualization
            outputs["images_gt"] = images
        else:
            B = len(data["prompt"])
            F, H, W = (self.opt.num_input_frames_test - 1) * actual_num_clips + 1, self.opt.input_res[0], self.opt.input_res[1]
            images = torch.zeros((B, F, 3, H, W), dtype=dtype, device=device)  # (B, F, 3, H, W); not really used

        idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
        images_f = images[:, idxs, ...]  # (B, f, 3, H, W)
        if self.opt.da3_down_ratio != 1:
            images_f = mv_interpolate(images_f,
                size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)
        if "C2W" in data and "fxfycxcy" in data:
            C2W = data["C2W"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 4, 4)
            fxfycxcy = data["fxfycxcy"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 4)
            plucker = plucker_ray(H, W, C2W.float(), fxfycxcy.float())[0].to(dtype)  # (B, f, 6, H, W)
        else:
            C2W, fxfycxcy, plucker = None, None, None
        if "depth" in data:
            depths = data["depth"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
        else:
            depths = None
        if "conf" in data:
            confs = data["conf"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
        else:
            confs = None

        f = 1 + (F - 1) // self.opt.compression_ratio[0]
        h = H // self.opt.compression_ratio[1]
        w = W // self.opt.compression_ratio[2]

        # Text encoder
        if self.text_encoder is not None:
            if self.prompt_list is None or np.random.rand() >= self.opt.vidprom_prob:
                prompts = data["prompt"]  # a list of strings
            else:
                assert actual_num_clips == 1  # VidProm only supports single clip
                prompts = np.random.choice(self.prompt_list, B, replace=False).tolist()
            self.text_encoder.eval()
            prompt_embeds = self._encode_prompt_batch(prompts)  # (B, N=512, D') or (B, num_clips, N=512, D')
            negative_prompt_embeds = self._build_negative_prompt_embeds(B, actual_num_clips)  # (B, N=512, D') or (B, num_clips, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if self.opt.first_latent_cond and "image" in data:
            cond_latents = self.encode(images[:, 0:1, ...] * 2. - 1., vae)  # (B, D, 1, h, w)
        else:
            cond_latents = None
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Point cloud rendering
        all_points, all_colors = [None] * B, [None] * B
        if self.opt.input_pcrender:
            assert depths is not None and confs is not None and C2W is not None and fxfycxcy is not None
            all_render_images, all_render_depths = [], []
            for i in range(B):
                points, colors = filter_da3_points(
                    images_f[i], depths[i], confs[i], C2W[i], fxfycxcy[i],
                    conf_thresh_percentile=self.opt.conf_thresh_percentile,
                    random_sample_ratio=self.opt.rand_pcrender_ratio,
                    min_num_points=self.opt.min_num_points,
                    max_num_points=self.opt.max_num_points,
                    all_valid=True,  # save all points for image conditioning
                )
                all_points[i], all_colors[i] = points, colors
                render_images, render_depths = render_pt3d_points(
                    H, W, points, colors, C2W[i], fxfycxcy[i],
                    return_depth=True,
                )  # (f, 3, H, W) in [0, 1]; (f, H, W)
                all_render_images.append(render_images.to(dtype))
                all_render_depths.append(render_depths.to(dtype))
            render_images = torch.stack(all_render_images, dim=0)  # (B, f, 3, H, W) in [0, 1]
            render_depths = torch.stack(all_render_depths, dim=0)  # (B, f, H, W)
        else:
            render_images, render_depths = None, None

        # (Optional) Extra conditioning: image + depth + mask
        if render_images is not None:
            input_extra_condition = torch.cat([
                render_images, render_depths.unsqueeze(2),
                (render_depths > 0.).unsqueeze(2).to(dtype),
            ], dim=2)  # (B, f, 3+1+1, H, W)
        else:
            input_extra_condition = None

        # Denoising
        for cfg_scale in self.opt.cfg_scale:
            latents = torch.randn(B, self.opt.latent_dim, f, h, w, device=device, dtype=dtype)
            if cond_latents is not None:
                latents = torch.cat([cond_latents, latents[:, :, 1:, ...]], dim=2)

            for ti, timestep in tqdm(enumerate(self.diffusion.scheduler.timesteps),
                total=len(self.diffusion.scheduler.timesteps), ncols=125, disable=not verbose, desc="[Denoise]"):
                timesteps = timestep * torch.ones(B, f).to(dtype=dtype, device=device)
                if cond_latents is not None:
                    timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

                ## Diffusion model
                model_outputs = self.diffusion(
                    latents,
                    timesteps,
                    prompt_embeds,
                    plucker=plucker if self.opt.input_plucker else None,
                    C2W=C2W, fxfycxcy=fxfycxcy,  # for DA3
                    extra_condition=input_extra_condition,
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                )

                ## CFG
                if cfg_scale > 1.:
                    model_outputs_neg = self.diffusion(
                        latents,
                        timesteps,
                        negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                        plucker=plucker if self.opt.input_plucker else None,
                        C2W=C2W, fxfycxcy=fxfycxcy,  # for DA3
                        extra_condition=input_extra_condition,
                        #
                        clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    )
                    if not self.opt.load_da3:
                        model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                    else:
                        model_outputs, da3_outputs = model_outputs
                        model_outputs_neg, _ = model_outputs_neg
                        model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                        model_outputs = (model_outputs, da3_outputs)

                model_outputs, da3_outputs = \
                    model_outputs if self.opt.load_da3 else (model_outputs, None)

                if self.opt.deterministic_inference:
                    latents = self.diffusion.scheduler.step(
                        model_outputs.transpose(1, 2).flatten(0, 1),
                        timesteps.flatten(0, 1),
                        latents.transpose(1, 2).flatten(0, 1),
                    ).unflatten(0, (B, f)).transpose(1, 2)  # (B, D, f, h, w)
                else:
                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, latents, timesteps)
                    if ti < len(self.diffusion.scheduler.timesteps) - 1:
                        next_timesteps = self.diffusion.scheduler.timesteps[ti + 1] * torch.ones_like(timesteps)
                        if cond_latents is not None:
                            next_timesteps = torch.cat([torch.zeros_like(next_timesteps[:, :1]), next_timesteps[:, 1:]], dim=1)

                        latents = self.diffusion.scheduler.add_noise(
                            pred_x0.transpose(1, 2).flatten(0, 1),
                            torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                            next_timesteps.flatten(0, 1),
                        ).unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)
                    else:
                        latents = pred_x0

            # Decode
            pred_images = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.  # (B, D, f, h, w) -> (B, F, 3, H, W)
            outputs[f"images_pred_{cfg_scale}"] = pred_images

            # Evaluation metrics: PSNR, SSIM, LPIPS
            if "image" in data:
                outputs[f"psnr_{cfg_scale}"] = -10. * torch.log10(torch.mean((images - pred_images) ** 2))  # (,)
                outputs[f"ssim_{cfg_scale}"] = SSIM(
                    rearrange(pred_images, "b f c h w -> (b f) c h w"),
                    rearrange(images, "b f c h w -> (b f) c h w"),
                    data_range=1., size_average=False,
                ).mean()  # (,)
                if self.lpips_loss is not None:
                    outputs[f"lpips_{cfg_scale}"] = self.lpips_loss(
                        rearrange(pred_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                        rearrange(images, "b f c h w -> (b f) c h w") * 2. - 1.,
                    ).mean()  # (,)

            # (Optional) DA3 evaluation
            if self.opt.load_da3:
                assert da3_outputs is not None
                assert depths is not None and C2W is not None and fxfycxcy is not None

                if depths is not None:
                    outputs[f"depth_{cfg_scale}"] = tF.mse_loss(da3_outputs["depth"], depths)  # (,)

                ## Get ground-truth geometry labels
                _, (ray_o, ray_d) = plucker_ray(H//2//self.opt.da3_down_ratio, W//2//self.opt.da3_down_ratio,
                    C2W.float(), fxfycxcy.float(), normalize_ray_d=False)
                gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
                gt_pose_enc = torch.cat([
                    C2W[:, :, :3, 3].float(),  # (B, f, 3)
                    mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
                    2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
                    2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
                ], dim=-1).to(dtype)  # (B, f, 9)

                ## Compute geometry metrics via MSE
                outputs[f"ray_{cfg_scale}"] = tF.mse_loss(da3_outputs["ray"], gt_raymaps)  # (,)
                outputs[f"pose_{cfg_scale}"] = tF.mse_loss(da3_outputs["pose_enc"], gt_pose_enc)  # (,)

                # For visualization
                outputs[f"images_pred_depth_{cfg_scale}"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)

        if self.opt.load_da3 and depths is not None:
            outputs["images_gt_depth"] = colorize_depth(1./depths, batch_mode=True)

        if render_images is not None:
            outputs["images_render"] = render_images

        return outputs

    def evaluate_causal(self, data: Dict[str, Any], dtype: torch.dtype, verbose: bool = True, vae: Optional[WanVAEWrapper] = None):
        outputs = {}
        device = self.diffusion.model.device

        # For multi-clip generation
        data, clip_latent_lens = self._multiclip_batch(data)
        # Determine actual num_clips from data
        actual_num_clips = len(data["prompt"][0]) if isinstance(data["prompt"][0], list) else 1
        if actual_num_clips > 1:
            assert clip_latent_lens is not None

        self.diffusion.eval()
        self.diffusion.scheduler.set_timesteps(self.opt.num_inference_steps, training=False)

        if "image" in data:
            images = data["image"].to(device=device, dtype=dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W) = images.shape

            # For visualization
            outputs["images_gt"] = images
        else:
            B = len(data["prompt"])
            F, H, W = (self.opt.num_input_frames_test - 1) * actual_num_clips + 1, self.opt.input_res[0], self.opt.input_res[1]
            images = torch.zeros((B, F, 3, H, W), dtype=dtype, device=device)  # (B, F, 3, H, W); not really used

        idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
        images_f = images[:, idxs, ...]  # (B, f, 3, H, W)
        if self.opt.da3_down_ratio != 1:
            images_f = mv_interpolate(images_f,
                size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)
        if "C2W" in data and "fxfycxcy" in data:
            C2W = data["C2W"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 4, 4)
            fxfycxcy = data["fxfycxcy"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 4)
            plucker = plucker_ray(H, W, C2W.float(), fxfycxcy.float())[0].to(dtype)  # (B, f, 6, H, W)
        else:
            C2W, fxfycxcy, plucker = None, None, None
        if "depth" in data:
            depths = data["depth"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
        else:
            depths = None
        if "conf" in data:
            confs = data["conf"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, H, W)
        else:
            confs = None

        f = 1 + (F - 1) // self.opt.compression_ratio[0]
        h = H // self.opt.compression_ratio[1]
        w = W // self.opt.compression_ratio[2]

        # Text encoder
        if self.text_encoder is not None:
            if self.prompt_list is None or np.random.rand() >= self.opt.vidprom_prob:
                prompts = data["prompt"]  # a list of strings
            else:
                assert actual_num_clips == 1  # VidProm only supports single clip
                prompts = np.random.choice(self.prompt_list, B, replace=False).tolist()
            self.text_encoder.eval()
            prompt_embeds = self._encode_prompt_batch(prompts)  # (B, N=512, D') or (B, num_clips, N=512, D')
            negative_prompt_embeds = self._build_negative_prompt_embeds(B, actual_num_clips)  # (B, N=512, D') or (B, num_clips, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if self.opt.first_latent_cond and "image" in data:
            cond_latents = self.encode(images[:, 0:1, ...] * 2. - 1., vae)  # (B, D, 1, h, w)
        else:
            cond_latents = None
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Point cloud rendering
        all_points, all_colors = [None] * B, [None] * B
        if self.opt.input_pcrender:
            if cond_latents is not None:
                assert depths is not None and confs is not None and C2W is not None and fxfycxcy is not None
                all_render_images, all_render_depths = [], []
                for i in range(B):
                    points, colors = filter_da3_points(
                        images[i, 0:1], depths[i, 0:1], confs[i, 0:1], C2W[i, 0:1], fxfycxcy[i, 0:1],
                        conf_thresh_percentile=self.opt.conf_thresh_percentile,
                        random_sample_ratio=self.opt.rand_pcrender_ratio,
                        min_num_points=self.opt.min_num_points,
                        max_num_points=self.opt.max_num_points,
                        all_valid=True,  # save all points for image conditioning
                    )
                    all_points[i], all_colors[i] = points, colors
                    all_render_images_chunk, all_render_depths_chunk = render_pt3d_points(
                        H, W, points, colors,
                        C2W[i, :self.opt.chunk_size], fxfycxcy[i, :self.opt.chunk_size],
                        return_depth=True,
                    )  # (f_chunk, 3, H, W) in [0, 1]; (f_chunk, H, W)
                    all_render_images.append(all_render_images_chunk.to(dtype))
                    all_render_depths.append(all_render_depths_chunk.to(dtype))
                all_render_images = torch.stack(all_render_images, dim=0)  # (B, f_chunk, 3, H, W) in [0, 1]
                all_render_depths = torch.stack(all_render_depths, dim=0)  # (B, f_chunk, H, W)
                render_images, render_depths = all_render_images, all_render_depths
            else:
                render_images = torch.zeros((B, self.opt.chunk_size, 3, H, W), dtype=dtype, device=device)
                render_depths = torch.zeros((B, self.opt.chunk_size, H, W), dtype=dtype, device=device)
            render_images_vis = render_images

            if self.opt.load_da3:
                if self.opt.load_tae:
                    vae_cache = None
                else:
                    vae_cache = {
                        512: ZERO_VAE_CACHE_512,
                        832: ZERO_VAE_CACHE,
                    }[W]
                    for i in range(len(vae_cache)):
                        vae_cache[i] = vae_cache[i].to(device=device, dtype=dtype)
        else:
            render_images, render_depths = None, None

        # (Optional) Extra conditioning: image + depth + mask
        if render_images is not None:
            input_extra_condition = torch.cat([
                render_images, render_depths.unsqueeze(2),
                (render_depths > 0.).unsqueeze(2).to(dtype),
            ], dim=2)  # (B, f_chunk, 3+1+1, H, W)
        else:
            input_extra_condition = None

        # Denoising
        for cfg_scale in self.opt.cfg_scale:
            latents = torch.randn(B, self.opt.latent_dim, f, h, w, device=device, dtype=dtype)
            if cond_latents is not None:
                latents = torch.cat([cond_latents, latents[:, :, 1:, ...]], dim=2)

            ## Set KV cache
            self._initialize_kv_cache(B, device=device, dtype=dtype)
            self._initialize_crossattn_cache(B, device=device, dtype=dtype)

            ## Auto-regression steps
            assert f % self.opt.chunk_size == 0
            num_chunks = f // self.opt.chunk_size
            frame_seqlen = h * w // 4  # `4`: hard-coded for 2x2 patch embedding in DiT

            ## Temporal denoising loop
            all_da3_outputs = [None] * num_chunks
            for chunk_idx in tqdm(range(num_chunks), ncols=125, disable=not verbose, desc="[Chunk]"):
                this_chunk_latents = latents[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
                if self.opt.input_plucker:
                    this_chunk_plucker = plucker[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
                else:
                    this_chunk_plucker = None
                if C2W is not None and fxfycxcy is not None:
                    this_chunk_C2W = C2W[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
                    this_chunk_fxfycxcy = fxfycxcy[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
                else:
                    this_chunk_C2W, this_chunk_fxfycxcy = None, None

                ### Spatial denoising loop
                for ti, timestep in tqdm(enumerate(self.diffusion.scheduler.timesteps),
                    total=len(self.diffusion.scheduler.timesteps), ncols=125, disable=not verbose, desc="[Denoise]"):
                    timesteps = timestep[None, None].repeat(B, self.opt.chunk_size).to(dtype=dtype, device=device)
                    if chunk_idx == 0 and cond_latents is not None:
                        timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

                    #### Diffusion model
                    model_outputs = self.diffusion(
                        this_chunk_latents,
                        timesteps,
                        prompt_embeds,
                        plucker=this_chunk_plucker,
                        C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                        extra_condition=input_extra_condition,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                        #
                        clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    )

                    #### CFG
                    if cfg_scale > 1.:
                        model_outputs_neg = self.diffusion(
                            this_chunk_latents,
                            timesteps,
                            negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                            plucker=this_chunk_plucker,
                            C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                            extra_condition=input_extra_condition,
                            #
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                            #
                            kv_cache_da3=self.kv_cache_neg_da3,
                            current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                            #
                            clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                        )

                        if not self.opt.load_da3:
                            model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                        else:
                            model_outputs, da3_outputs = model_outputs
                            model_outputs_neg, _ = model_outputs_neg
                            model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                            model_outputs = (model_outputs, da3_outputs)

                    model_outputs, da3_outputs = \
                        model_outputs if self.opt.load_da3 else (model_outputs, None)

                    if self.opt.deterministic_inference:
                        this_chunk_latents = self.diffusion.scheduler.step(
                            model_outputs.transpose(1, 2).flatten(0, 1),
                            timesteps.flatten(0, 1),
                            this_chunk_latents.transpose(1, 2).flatten(0, 1),
                        ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
                    else:
                        pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps)
                        if ti < len(self.diffusion.scheduler.timesteps) - 1:
                            next_timesteps = self.diffusion.scheduler.timesteps[ti + 1] * torch.ones_like(timesteps)
                            if chunk_idx == 0 and cond_latents is not None:
                                next_timesteps = torch.cat([torch.zeros_like(next_timesteps[:, :1]), next_timesteps[:, 1:]], dim=1)

                            this_chunk_latents = self.diffusion.scheduler.add_noise(
                                pred_x0.transpose(1, 2).flatten(0, 1),
                                torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                                next_timesteps.flatten(0, 1),
                            ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)  # (B, D, f_chunk, h, w)
                        else:
                            this_chunk_latents = pred_x0

                # Rerun with timestep zero to update KV cache
                # TODO: add noise on KV cache, except the first chunk
                model_outputs = self.diffusion(
                    this_chunk_latents,
                    timesteps * 0.,
                    prompt_embeds,
                    plucker=this_chunk_plucker,
                    C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                    extra_condition=input_extra_condition,
                    #
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                    #
                    kv_cache_da3=self.kv_cache_pos_da3,
                    current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                )

                if cfg_scale > 1.:
                    model_outputs_neg = self.diffusion(
                        this_chunk_latents,
                        timesteps * 0.,
                        negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                        plucker=this_chunk_plucker,
                        C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                        extra_condition=input_extra_condition,
                        #
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        kv_cache_da3=self.kv_cache_neg_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                        #
                        clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    )

                    if not self.opt.load_da3:
                        model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                    else:
                        model_outputs, da3_outputs = model_outputs
                        model_outputs_neg, _ = model_outputs_neg
                        model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                        model_outputs = (model_outputs, da3_outputs)

                model_outputs, da3_outputs = \
                    model_outputs if self.opt.load_da3 else (model_outputs, None)
                all_da3_outputs[chunk_idx] = da3_outputs

                if self.opt.deterministic_inference:
                    this_chunk_latents = self.diffusion.scheduler.step(
                        model_outputs.transpose(1, 2).flatten(0, 1),
                        timesteps.flatten(0, 1) * 0.,
                        this_chunk_latents.transpose(1, 2).flatten(0, 1),
                    ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
                else:
                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps * 0.)
                    this_chunk_latents = pred_x0

                # Record this chunk generated latents
                latents[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...] = this_chunk_latents

                # (Optional) Update render images for next chunks
                if self.opt.input_pcrender and chunk_idx < num_chunks - 1:
                    if self.opt.load_da3:
                        assert self.current_vae_decoder is not None

                        if self.opt.load_tae:
                            if vae_cache is None:
                                vae_cache = this_chunk_latents
                            else:
                                this_chunk_latents = torch.cat([vae_cache, this_chunk_latents], dim=2)
                                vae_cache = this_chunk_latents[:, :, -3:, :, :]
                            current_images_f = self.current_vae_decoder.decode(this_chunk_latents)
                            if chunk_idx == 0:
                                current_images_f = current_images_f[:, 3:, :, :, :]  # skip the first 3 frames of first block
                            else:
                                current_images_f = current_images_f[:, 12:, :, :, :]
                        else:
                            current_images_f, vae_cache = self.current_vae_decoder(this_chunk_latents, *vae_cache)
                            current_images_f = (current_images_f.clamp(-1., 1.) + 1.) / 2.
                            current_images_f = current_images_f.transpose(1, 2)  # (B, f', 3, H, W)
                            if chunk_idx == 0:
                                current_images_f = current_images_f[:, 3:, :, :, :]  # skip the first 3 frames of first block
                        if chunk_idx == 0:
                            _idxs = torch.arange(0, current_images_f.shape[1], 4).to(device=device, dtype=torch.long)
                        else:
                            _idxs = torch.arange(3, current_images_f.shape[1], 4).to(device=device, dtype=torch.long)
                        current_images_f = current_images_f[:, _idxs, :, :, :]  # (B, f_chunk, 3, H, W)
                        assert current_images_f.shape[1] == self.opt.chunk_size
                        if self.opt.da3_down_ratio != 1:
                            current_images_f = mv_interpolate(current_images_f,
                                size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)

                        current_depths = all_da3_outputs[chunk_idx]["depth"]  # (B, f_chunk, H, W)
                        current_confs = all_da3_outputs[chunk_idx]["depth_conf"]  # (B, f_chunk, H, W)
                        current_C2W = all_da3_outputs[chunk_idx]["C2W"]  # (B, f_chunk, 4, 4)
                        current_fxfycxcy = all_da3_outputs[chunk_idx]["fxfycxcy"]  # (B, f_chunk, 4)
                    else:  # use ground-truth as conditions
                        assert depths is not None and confs is not None and C2W is not None and fxfycxcy is not None
                        chunk_idxs = torch.arange(chunk_idx * self.opt.chunk_size, (chunk_idx + 1) * self.opt.chunk_size).to(device=device, dtype=torch.long)

                        current_images_f = images_f[:, chunk_idxs, ...]  # (B, f_chunk, 3, H, W)
                        current_depths = depths[:, chunk_idxs, ...]  # (B, f_chunk, H, W)
                        current_confs = confs[:, chunk_idxs, ...]  # (B, f_chunk, H, W)
                        current_C2W = C2W[:, chunk_idxs, ...]  # (B, f_chunk, 4, 4)
                        current_fxfycxcy = fxfycxcy[:, chunk_idxs, ...]  # (B, f_chunk, 4)

                    all_render_images, all_render_depths = [], []
                    for i in range(B):
                        points, colors = filter_da3_points(
                            current_images_f[i], current_depths[i], current_confs[i], current_C2W[i], current_fxfycxcy[i],
                            conf_thresh_percentile=self.opt.conf_thresh_percentile,
                            random_sample_ratio=self.opt.rand_pcrender_ratio,
                            min_num_points=self.opt.min_num_points,
                            max_num_points=self.opt.max_num_points,
                        )
                        if points.shape[0] > 0:
                            if all_points[i] is None:
                                all_points[i], all_colors[i] = points, colors
                            else:
                                all_points[i] = torch.cat([all_points[i], points], dim=0)
                                all_colors[i] = torch.cat([all_colors[i], colors], dim=0)
                            render_images, render_depths = render_pt3d_points(
                                H, W, all_points[i], all_colors[i],
                                C2W[i, (chunk_idx + 1) * self.opt.chunk_size:(chunk_idx + 2) * self.opt.chunk_size, ...],
                                fxfycxcy[i, (chunk_idx + 1) * self.opt.chunk_size:(chunk_idx + 2) * self.opt.chunk_size, ...],
                                return_depth=True,
                            )
                        else:  # no valid points
                            render_images = torch.zeros((self.opt.chunk_size, 3, H, W), dtype=dtype, device=device)
                            render_depths = torch.zeros((self.opt.chunk_size, H, W), dtype=dtype, device=device)
                        all_render_images.append(render_images.to(dtype))
                        all_render_depths.append(render_depths.to(dtype))
                    render_images = torch.stack(all_render_images, dim=0)  # (B, f_chunk, 3, H, W)
                    render_depths = torch.stack(all_render_depths, dim=0)  # (B, f_chunk, H, W)
                    render_images_vis = torch.cat([render_images_vis, render_images], dim=1)

                    # (Optional) Extra conditioning: image + depth + mask
                    input_extra_condition = torch.cat([
                        render_images, render_depths.unsqueeze(2),
                        (render_depths > 0.).unsqueeze(2).to(dtype),
                    ], dim=2)  # (B, f_chunk, 3+1+1, H, W)

            # Decode
            pred_images = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.  # (B, D, f, h, w) -> (B, F, 3, H, W)
            outputs[f"images_pred_{cfg_scale}"] = pred_images

            # Evaluation metrics: PSNR, SSIM, LPIPS
            if "image" in data:
                outputs[f"psnr_{cfg_scale}"] = -10. * torch.log10(torch.mean((images - pred_images) ** 2))  # (,)
                outputs[f"ssim_{cfg_scale}"] = SSIM(
                    rearrange(pred_images, "b f c h w -> (b f) c h w"),
                    rearrange(images, "b f c h w -> (b f) c h w"),
                    data_range=1., size_average=False,
                ).mean()  # (,)
                if self.lpips_loss is not None:
                    outputs[f"lpips_{cfg_scale}"] = self.lpips_loss(
                        rearrange(pred_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                        rearrange(images, "b f c h w -> (b f) c h w") * 2. - 1.,
                    ).mean()  # (,)

            # (Optional) DA3 evaluation
            if self.opt.load_da3:
                assert da3_outputs is not None
                assert depths is not None and C2W is not None and fxfycxcy is not None
                da3_outputs = {
                    k: torch.cat([all_da3_outputs[i][k] for i in range(num_chunks)], dim=1)
                    for k in all_da3_outputs[0].keys()
                }

                if depths is not None:
                    outputs[f"depth_{cfg_scale}"] = tF.mse_loss(da3_outputs["depth"], depths)  # (,)

                ## Get ground-truth geometry labels
                _, (ray_o, ray_d) = plucker_ray(H//2//self.opt.da3_down_ratio, W//2//self.opt.da3_down_ratio,
                    C2W.float(), fxfycxcy.float(), normalize_ray_d=False)
                gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
                gt_pose_enc = torch.cat([
                    C2W[:, :, :3, 3].float(),  # (B, f, 3)
                    mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
                    2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
                    2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
                ], dim=-1).to(dtype)  # (B, f, 9)

                ## Compute geometry metrics via MSE
                outputs[f"ray_{cfg_scale}"] = tF.mse_loss(da3_outputs["ray"], gt_raymaps)  # (,)
                outputs[f"pose_{cfg_scale}"] = tF.mse_loss(da3_outputs["pose_enc"], gt_pose_enc)  # (,)

                # For visualization
                outputs[f"images_pred_depth_{cfg_scale}"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)

            if render_images is not None:
                outputs[f"images_render_{cfg_scale}"] = render_images_vis

        if self.opt.load_da3 and depths is not None:
            outputs["images_gt_depth"] = colorize_depth(1./depths, batch_mode=True)

        return outputs


    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def infer(self,
        prompts: List[str],
        num_frames: int,
        #
        C2W: Optional[Tensor] = None,  # (B, f, 4, 4)
        fxfycxcy: Optional[Tensor] = None,  # (B, f, 4)
        #
        images: Optional[Tensor] = None,  # (B, 1, 3, H, W)
        depths: Optional[Tensor] = None,  # (B, 1, H, W)
        confs: Optional[Tensor] = None,  # (B, 1, H, W)
        #
        cfg_scale: Optional[float] = None,
        #
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = True,
        vae: Optional[WanVAEWrapper] = None,
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
    ):
        outputs = {}

        self.diffusion.eval()
        self.diffusion.scheduler.set_timesteps(self.opt.num_inference_steps, training=False)

        if cfg_scale is None:
            cfg_scale = self.opt.cfg_scale[-1]

        B = len(prompts)
        F, H, W = num_frames, self.opt.input_res[0], self.opt.input_res[1]
        device = self.diffusion.model.device

        f = 1 + (F - 1) // self.opt.compression_ratio[0]
        h = H // self.opt.compression_ratio[1]
        w = W // self.opt.compression_ratio[2]

        if C2W is not None and fxfycxcy is not None:
            assert C2W.shape[1] == f and fxfycxcy.shape[1] == f
            C2W = C2W.to(dtype)  # (B, f, 4, 4)
            fxfycxcy = fxfycxcy.to(dtype)  # (B, f, 4)
            plucker = plucker_ray(H, W, C2W.float(), fxfycxcy.float())[0].to(dtype)  # (B, f, 6, H, W)
        else:
            C2W, fxfycxcy, plucker = None, None, None
        if depths is not None:
            depths = depths.to(dtype)  # (B, 1, H, W)
        if confs is not None:
            confs = confs.to(dtype)  # (B, 1, H, W)

        if not self.opt.is_causal:
            assert self.opt.chunk_size == f  # for non-causal model, chunk_size should cover the full frame length

        # Text encoder
        if self.text_encoder is not None:
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
            negative_prompt_embeds = self.text_encoder([self.opt.negative_prompt]).repeat(B, 1, 1)  # (B, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if images is not None:
            images = images.to(dtype)  # (B, 1, 3, H, W)
            if self.opt.prefill_image:
                cond_latents = self.encode(images.repeat(1, 1 + 4 * (self.opt.chunk_size - 1), 1, 1, 1) * 2. - 1., vae)  # (B, D, f_chunk, h, w)
                assert cond_latents.shape[2] == self.opt.chunk_size
            else:
                cond_latents = self.encode(images * 2. - 1., vae)  # (B, D, 1, h, w)
        else:
            cond_latents = None

        # (Optional) Point cloud rendering
        all_points, all_colors = [None] * B, [None] * B
        if self.opt.input_pcrender:
            if cond_latents is not None:
                assert depths is not None and confs is not None and C2W is not None and fxfycxcy is not None
                all_render_images, all_render_depths = [], []
                for i in range(B):
                    points, colors = filter_da3_points(
                        images[i, 0:1], depths[i, 0:1], confs[i, 0:1], C2W[i, 0:1], fxfycxcy[i, 0:1],
                        conf_thresh_percentile=self.opt.conf_thresh_percentile,
                        random_sample_ratio=self.opt.rand_pcrender_ratio,
                        min_num_points=self.opt.min_num_points,
                        max_num_points=self.opt.max_num_points,
                        all_valid=True,  # save all points for image conditioning
                    )
                    all_points[i], all_colors[i] = points, colors
                    all_render_images_chunk, all_render_depths_chunk = render_pt3d_points(
                        H, W, points, colors,
                        C2W[i, :self.opt.chunk_size], fxfycxcy[i, :self.opt.chunk_size],
                        return_depth=True,
                    )  # (f_chunk, 3, H, W) in [0, 1]; (f_chunk, H, W)
                    all_render_images.append(all_render_images_chunk.to(dtype))
                    all_render_depths.append(all_render_depths_chunk.to(dtype))
                render_images = torch.stack(all_render_images, dim=0)  # (B, f_chunk, 3, H, W)
                render_depths = torch.stack(all_render_depths, dim=0)  # (B, f_chunk, H, W)
            else:
                render_images = torch.zeros((B, self.opt.chunk_size, 3, H, W), dtype=dtype, device=device)
                render_depths = torch.zeros((B, self.opt.chunk_size, H, W), dtype=dtype, device=device)
            render_images_vis = render_images

            if self.opt.load_da3:
                if self.opt.load_tae:
                    vae_cache = None
                else:
                    vae_cache = {
                        512: ZERO_VAE_CACHE_512,
                        832: ZERO_VAE_CACHE,
                    }[W]
                    for i in range(len(vae_cache)):
                        vae_cache[i] = vae_cache[i].to(device=device, dtype=dtype)
        else:
            render_images, render_depths = None, None

        # (Optional) Extra conditioning: image + depth + mask
        if render_images is not None:
            input_extra_condition = torch.cat([
                render_images, render_depths.unsqueeze(2),
                (render_depths > 0.).unsqueeze(2).to(dtype),
            ], dim=2)  # (B, f, 3+1+1, H, W)
        else:
            input_extra_condition = None

        # Denoising
        latents = torch.randn(B, self.opt.latent_dim, f, h, w, device=device, dtype=dtype)
        if not self.opt.prefill_image and cond_latents is not None:
            latents = torch.cat([cond_latents, latents[:, :, 1:, ...]], dim=2)

        ## Set KV cache
        self._initialize_kv_cache(B, device=device, dtype=dtype)
        self._initialize_crossattn_cache(B, device=device, dtype=dtype)

        ## Auto-regression steps
        assert f % self.opt.chunk_size == 0
        num_chunks = f // self.opt.chunk_size
        frame_seqlen = h * w // 4  # `4`: hard-coded for 2x2 patch embedding in DiT

        # (Optional) Prefill cache with cond_latents
        if self.opt.prefill_image and cond_latents is not None:
            self.diffusion(
                cond_latents,
                torch.zeros((B, self.opt.chunk_size), device=device, dtype=dtype),
                prompt_embeds,
                plucker=plucker[:, 0:1, ...].repeat(1, self.opt.chunk_size, 1, 1, 1) if plucker is not None else None,
                C2W=C2W[:, 0:1, ...].repeat(1, self.opt.chunk_size, 1, 1) if C2W is not None else None,  # for DA3
                fxfycxcy=fxfycxcy[:, 0:1, ...].repeat(1, self.opt.chunk_size, 1) if fxfycxcy is not None else None,  # for DA3
                extra_condition=torch.zeros_like(input_extra_condition) \
                    if input_extra_condition is not None else None,  # align with the original t2v first chunk conditioning
                #
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=0,
                #
                kv_cache_da3=self.kv_cache_pos_da3,
                current_start_da3=0,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
            )
            if cfg_scale > 1.:
                self.diffusion(
                    cond_latents,
                    torch.zeros((B, self.opt.chunk_size), device=device, dtype=dtype),
                    negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                    plucker=plucker[:, 0:1, ...].repeat(1, self.opt.chunk_size, 1, 1, 1) if plucker is not None else None,
                    C2W=C2W[:, 0:1, ...].repeat(1, self.opt.chunk_size, 1, 1) if C2W is not None else None,  # for DA3
                    fxfycxcy=fxfycxcy[:, 0:1, ...].repeat(1, self.opt.chunk_size, 1) if fxfycxcy is not None else None,  # for DA3
                    extra_condition=torch.zeros_like(input_extra_condition) \
                        if input_extra_condition is not None else None,  # align with the original t2v first chunk conditioning
                    #
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=0,
                    #
                    kv_cache_da3=self.kv_cache_neg_da3,
                    current_start_da3=0,
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                )
            cache_start_chunk_idx = 1
        else:
            cache_start_chunk_idx = 0

        ## Temporal denoising loop
        all_da3_outputs = [None] * num_chunks
        for chunk_idx in tqdm(range(num_chunks), ncols=125, disable=not verbose, desc="[Chunk]"):
            this_chunk_latents = latents[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            if self.opt.input_plucker:
                this_chunk_plucker = plucker[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            else:
                this_chunk_plucker = None
            if C2W is not None and fxfycxcy is not None:
                this_chunk_C2W = C2W[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
                this_chunk_fxfycxcy = fxfycxcy[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            else:
                this_chunk_C2W, this_chunk_fxfycxcy = None, None

            ### Spatial denoising loop
            for ti, timestep in tqdm(enumerate(self.diffusion.scheduler.timesteps),
                total=len(self.diffusion.scheduler.timesteps), ncols=125, disable=not verbose, desc="[Denoise]"):
                timesteps = timestep[None, None].repeat(B, self.opt.chunk_size).to(dtype=dtype, device=device)
                if not self.opt.prefill_image and chunk_idx == 0 and cond_latents is not None:
                    timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

                #### Diffusion model
                model_outputs = self.diffusion(
                    this_chunk_latents,
                    timesteps,
                    prompt_embeds,
                    plucker=this_chunk_plucker,
                    C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                    extra_condition=input_extra_condition,
                    #
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * frame_seqlen,
                    #
                    kv_cache_da3=self.kv_cache_pos_da3,
                    current_start_da3=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                )

                #### CFG
                if cfg_scale > 1.:
                    model_outputs_neg = self.diffusion(
                        this_chunk_latents,
                        timesteps,
                        negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                        plucker=this_chunk_plucker,
                        C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                        extra_condition=input_extra_condition,
                        #
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * frame_seqlen,
                        #
                        kv_cache_da3=self.kv_cache_neg_da3,
                        current_start_da3=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                        #
                        clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    )

                    if not self.opt.load_da3:
                        model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                    else:
                        model_outputs, da3_outputs = model_outputs
                        model_outputs_neg, _ = model_outputs_neg
                        model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                        model_outputs = (model_outputs, da3_outputs)

                model_outputs, da3_outputs = \
                    model_outputs if self.opt.load_da3 else (model_outputs, None)

                if self.opt.deterministic_inference:
                    this_chunk_latents = self.diffusion.scheduler.step(
                        model_outputs.transpose(1, 2).flatten(0, 1),
                        timesteps.flatten(0, 1),
                        this_chunk_latents.transpose(1, 2).flatten(0, 1),
                    ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
                else:
                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps)
                    if ti < len(self.diffusion.scheduler.timesteps) - 1:
                        next_timesteps = self.diffusion.scheduler.timesteps[ti + 1] * torch.ones_like(timesteps)
                        if not self.opt.prefill_image and chunk_idx == 0 and cond_latents is not None:
                            next_timesteps = torch.cat([torch.zeros_like(next_timesteps[:, :1]), next_timesteps[:, 1:]], dim=1)

                        this_chunk_latents = self.diffusion.scheduler.add_noise(
                            pred_x0.transpose(1, 2).flatten(0, 1),
                            torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                            next_timesteps.flatten(0, 1),
                        ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)  # (B, D, f_chunk, h, w)
                    else:
                        this_chunk_latents = pred_x0

            # Rerun with timestep zero to update KV cache
            # TODO: add noise on KV cache, except the first chunk
            model_outputs = self.diffusion(
                this_chunk_latents,
                timesteps * 0.,
                prompt_embeds,
                plucker=this_chunk_plucker,
                C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                extra_condition=input_extra_condition,
                #
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * frame_seqlen,
                #
                kv_cache_da3=self.kv_cache_pos_da3,
                current_start_da3=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
            )

            if cfg_scale > 1.:
                model_outputs_neg = self.diffusion(
                    this_chunk_latents,
                    timesteps * 0.,
                    negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                    plucker=this_chunk_plucker,
                    C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                    extra_condition=input_extra_condition,
                    #
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * frame_seqlen,
                    #
                    kv_cache_da3=self.kv_cache_neg_da3,
                    current_start_da3=(cache_start_chunk_idx + chunk_idx) * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                )

                if not self.opt.load_da3:
                    model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                else:
                    model_outputs, da3_outputs = model_outputs
                    model_outputs_neg, _ = model_outputs_neg
                    model_outputs = model_outputs_neg + cfg_scale * (model_outputs - model_outputs_neg)
                    model_outputs = (model_outputs, da3_outputs)

            model_outputs, da3_outputs = \
                model_outputs if self.opt.load_da3 else (model_outputs, None)
            all_da3_outputs[chunk_idx] = da3_outputs

            if self.opt.deterministic_inference:
                this_chunk_latents = self.diffusion.scheduler.step(
                    model_outputs.transpose(1, 2).flatten(0, 1),
                    timesteps.flatten(0, 1) * 0.,
                    this_chunk_latents.transpose(1, 2).flatten(0, 1),
                ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
            else:
                pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps * 0.)
                this_chunk_latents = pred_x0

            # Record this chunk generated latents
            latents[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...] = this_chunk_latents

            # (Optional) Update render images for next chunks
            if self.opt.input_pcrender and chunk_idx < num_chunks - 1:
                assert self.opt.load_da3
                assert self.current_vae_decoder is not None

                if self.opt.load_tae:
                    if vae_cache is None:
                        vae_cache = this_chunk_latents
                    else:
                        this_chunk_latents = torch.cat([vae_cache, this_chunk_latents], dim=2)
                        vae_cache = this_chunk_latents[:, :, -3:, :, :]
                    current_images_f = self.current_vae_decoder.decode(this_chunk_latents)
                    if chunk_idx == 0:
                        current_images_f = current_images_f[:, 3:, :, :, :]  # skip the first 3 frames of first block
                    else:
                        current_images_f = current_images_f[:, 12:, :, :, :]
                else:
                    current_images_f, vae_cache = self.current_vae_decoder(this_chunk_latents, *vae_cache)
                    current_images_f = (current_images_f.clamp(-1., 1.) + 1.) / 2.
                    current_images_f = current_images_f.transpose(1, 2)  # (B, f', 3, H, W)
                    if chunk_idx == 0:
                        current_images_f = current_images_f[:, 3:, :, :, :]  # skip the first 3 frames of first block
                if chunk_idx == 0:
                    _idxs = torch.arange(0, current_images_f.shape[1], 4).to(device=device, dtype=torch.long)
                else:
                    _idxs = torch.arange(3, current_images_f.shape[1], 4).to(device=device, dtype=torch.long)
                current_images_f = current_images_f[:, _idxs, :, :, :]  # (B, f_chunk, 3, H, W)
                assert current_images_f.shape[1] == self.opt.chunk_size
                if self.opt.da3_down_ratio != 1:
                    current_images_f = mv_interpolate(current_images_f,
                        size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)

                current_depths = all_da3_outputs[chunk_idx]["depth"]  # (B, f_chunk, H, W)
                current_confs = all_da3_outputs[chunk_idx]["depth_conf"]  # (B, f_chunk, H, W)
                current_C2W = all_da3_outputs[chunk_idx]["C2W"]  # (B, f_chunk, 4, 4)
                current_fxfycxcy = all_da3_outputs[chunk_idx]["fxfycxcy"]  # (B, f_chunk, 4)

                all_render_images, all_render_depths = [], []
                for i in range(B):
                    points, colors = filter_da3_points(
                        current_images_f[i], current_depths[i], current_confs[i], current_C2W[i], current_fxfycxcy[i],
                        conf_thresh_percentile=self.opt.conf_thresh_percentile,
                        random_sample_ratio=self.opt.rand_pcrender_ratio,
                        min_num_points=self.opt.min_num_points,
                        max_num_points=self.opt.max_num_points,
                    )
                    if points.shape[0] > 0:
                        if all_points[i] is None:
                            all_points[i], all_colors[i] = points, colors
                        else:
                            all_points[i] = torch.cat([all_points[i], points], dim=0)
                            all_colors[i] = torch.cat([all_colors[i], colors], dim=0)
                        render_images, render_depths = render_pt3d_points(
                            H, W, all_points[i], all_colors[i],
                            C2W[i, (chunk_idx + 1) * self.opt.chunk_size:(chunk_idx + 2) * self.opt.chunk_size, ...],
                            fxfycxcy[i, (chunk_idx + 1) * self.opt.chunk_size:(chunk_idx + 2) * self.opt.chunk_size, ...],
                            return_depth=True,
                        )
                    else:  # no valid points
                        render_images = torch.zeros((self.opt.chunk_size, 3, H, W), dtype=dtype, device=device)
                        render_depths = torch.zeros((self.opt.chunk_size, H, W), dtype=dtype, device=device)
                    all_render_images.append(render_images.to(dtype))
                    all_render_depths.append(render_depths.to(dtype))
                render_images = torch.stack(all_render_images, dim=0)  # (B, f_chunk, 3, H, W)
                render_depths = torch.stack(all_render_depths, dim=0)  # (B, f_chunk, H, W)
                render_images_vis = torch.cat([render_images_vis, render_images], dim=1)

                # (Optional) Extra conditioning: image + depth + mask
                input_extra_condition = torch.cat([
                    render_images, render_depths.unsqueeze(2),
                    (render_depths > 0.).unsqueeze(2).to(dtype),
                ], dim=2)  # (B, f_chunk, 3+1+1, H, W)

        # Decode
        if self.opt.prefill_image and cond_latents is not None:
            latents = torch.cat([cond_latents, latents], dim=2)
        pred_images = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.  # (B, D, f, h, w) -> (B, F, 3, H, W)
        # pred_images = torch.cat([images, pred_images[:, 1 + 4 * cache_start_chunk_idx * self.opt.chunk_size:, :, :, :]], dim=1)
        pred_images = pred_images[:, 4 * cache_start_chunk_idx * self.opt.chunk_size:, :, :, :]
        assert pred_images.shape[1] == F
        outputs["images_pred"] = pred_images

        if render_images is not None:
            outputs["images_render"] = render_images_vis

        return outputs


    ################################ Helper functions ################################


    def _multiclip_batch(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Tensor]]:
        prompts = data.get("prompt", None)
        if not isinstance(prompts[0], list):  # one-clip
            return data, None

        # Support dynamic num_clips: only support batch_size=1
        assert len(prompts) == 1 and len(prompts[0]) <= self.opt.num_clips

        new_data = dict(data)
        clip_frame_lens = []
        for key in ["image", "C2W", "fxfycxcy", "depth", "conf"]:
            if key not in data:
                continue
            value = data[key]  # a list (batch) of tuple (clip)
            clip_frame_lens = [[clip.shape[0] for clip in sample] for sample in value]
            new_data[key] = torch.stack([torch.cat(sample, dim=0) for sample in value], dim=0)  # (B=1, sum(F_clip), ...)

        clip_frame_lens = torch.tensor(clip_frame_lens, dtype=torch.long)  # (B=1, num_clips)
        clip_latent_lens = []
        for i in range(clip_frame_lens.shape[1]):
            if i == 0:  # first clip keeps the first image latent
                clip_latent_len = 1 + int(round((clip_frame_lens[0, 0:1].item() - 1) / self.opt.compression_ratio[0]))
                clip_latent_lens.append(clip_latent_len)
            elif i == clip_frame_lens.shape[1] - 1:  # last clip
                all_latent_len = 1 + int(round((clip_frame_lens[0, :].sum().item() - 1) / self.opt.compression_ratio[0]))
                clip_latent_len = all_latent_len - sum(clip_latent_lens)  # the last clip takes all remaining latents
                clip_latent_lens.append(clip_latent_len)
            else:  # middle clips
                clip_latent_len = int(round(clip_frame_lens[0, i:i+1].sum().item() / self.opt.compression_ratio[0]))
                clip_latent_lens.append(clip_latent_len)
        clip_latent_lens = torch.tensor(clip_latent_lens, dtype=torch.long)[None, :]  # (B=1, num_clips)

        return new_data, clip_latent_lens

    def _encode_prompt_batch(self, prompts: List[str] | List[List[str]]) -> Tensor:
        if len(prompts) == 0:
            raise ValueError("Prompts should not be empty")
        if isinstance(prompts[0], list):
            B, num_clips = len(prompts), len(prompts[0])
            flat_prompts = [p for sample in prompts for p in sample]
            embeds = self.text_encoder(flat_prompts)
            return embeds.reshape(B, num_clips, embeds.shape[1], embeds.shape[2])  # (B, num_clips, N=512, D')
        return self.text_encoder(prompts)  # (B, N=512, D')

    def _build_negative_prompt_embeds(self, batch_size: int, num_clips: int = 1) -> Tensor:
        neg = self.text_encoder([self.opt.negative_prompt]).repeat(batch_size * num_clips, 1, 1)
        if num_clips > 1:
            neg = neg.reshape(batch_size, num_clips, neg.shape[1], neg.shape[2])
        return neg  # (B, N=512, D') or (B, num_clips, N=512, D')

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """
        Initialize a per-GPU KV cache for the Wan model.
        """
        num_blocks = len(self.diffusion.model.blocks)
        num_heads = self.diffusion.model.num_heads
        head_dim = self.diffusion.model.dim // num_heads

        # When SP is active, KV cache is stored head-sharded
        sp_size = get_sp_world_size()
        num_heads_per_rank = num_heads // sp_size

        kv_cache_pos, kv_cache_neg = [], []
        for _ in range(num_blocks):
            kv_cache_pos.append({
                "k": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads_per_rank, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads_per_rank, head_dim), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
            kv_cache_neg.append({
                "k": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads_per_rank, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads_per_rank, head_dim), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

        if self.opt.load_da3:
            num_da3_blocks = len(self.diffusion.da3_model.backbone.pretrained.blocks)
            num_heads_da3 = self.diffusion.da3_model.backbone.pretrained.num_heads
            head_dim_da3 = self.diffusion.da3_model.backbone.pretrained.embed_dim // num_heads_da3

            # When SP is active, KV cache is stored head-sharded
            num_heads_da3_per_rank = num_heads_da3 // sp_size

            kv_cache_pos_da3, kv_cache_neg_da3 = [], []
            for _ in range(num_da3_blocks):
                kv_cache_pos_da3.append({
                    "k": torch.zeros((batch_size, num_heads_da3_per_rank, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, num_heads_da3_per_rank, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                })
                kv_cache_neg_da3.append({
                    "k": torch.zeros((batch_size, num_heads_da3_per_rank, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, num_heads_da3_per_rank, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                })
            self.kv_cache_pos_da3 = kv_cache_pos_da3  # always store the clean cache
            self.kv_cache_neg_da3 = kv_cache_neg_da3  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """
        Initialize a per-GPU cross-attention cache for the Wan model.
        """
        num_blocks = len(self.diffusion.model.blocks)
        num_heads = self.diffusion.model.num_heads
        head_dim = self.diffusion.model.dim // num_heads

        crossattn_cache_pos, crossattn_cache_neg = [], []
        for _ in range(num_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),  # `512` is hard-coded here (max_text_len)
                "v": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),
                "is_init": False,
            })
            crossattn_cache_neg.append({
                "k": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),  # `512` is hard-coded here (max_text_len)
                "v": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    @torch.no_grad()
    def encode(self, images: Tensor, vae: WanVAEWrapper):
        """ Image to VAE latent.

        Inputs:
            - `images`: (B, F, 3, H, W) in [-1, 1]

        Outputs:
            - `latents`: (B, D, f, h, w)
        """
        vae.eval()
        images = rearrange(images, "b f c h w -> b c f h w")  # (B, F, 3, H, W) -> (B, 3, F, H, W)
        return vae.encode(images).to(images.dtype)  # (B, D, f, h, w)

    @torch.no_grad()
    def decode_latent(self, latents: Tensor, vae: WanVAEWrapper):
        """ VAE latent to Image.

        Inputs:
            - `latents`: (B, D, f, h, w)

        Outputs:
            - `images`: (B, F, 3, H, W) in [-1, 1]
        """
        vae.eval()
        images = vae.decode(latents)  # (B, D, f, h, w) -> (B, 3, F, H, W)
        return rearrange(images, "b c f h w -> b f c h w").to(latents.dtype)  # (B, F, 3, H, W)

    def _add_lora_to_wan(self, target_modules: List[str], lora_rank: int, lora_alpha: Optional[int] = None):
        if lora_alpha is None:
            lora_alpha = lora_rank

        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        self.diffusion = inject_adapter_in_model(lora_config, self.diffusion)

        # Freeze all base model parameters, only train LoRA weights
        for name, param in self.diffusion.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def get_lora_state_dict(self):
        """Get only LoRA parameters for saving.

        This method is FSDP-aware and will gather sharded parameters from all ranks.
        When using FSDP, this should be called on all ranks, but only rank 0 will receive the full state dict.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Check if the model is wrapped with FSDP
        if isinstance(self.diffusion, FSDP):
            # Use FSDP API to gather full state dict on rank 0
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.diffusion, StateDictType.FULL_STATE_DICT, cfg):
                full_state_dict = self.diffusion.state_dict()

            # Filter for LoRA parameters
            lora_state_dict = {}
            for name, param in full_state_dict.items():
                # Remove FSDP wrapper prefixes
                clean_name = name.replace("_fsdp_wrapped_module.", "") \
                                .replace("_checkpoint_wrapped_module.", "") \
                                .replace("_orig_mod.", "") \
                                .replace("module.", "")
                if "lora_" in clean_name:
                    lora_state_dict[clean_name] = param
            return lora_state_dict
        else:
            # Non-FSDP case: directly access parameters
            lora_state_dict = {}
            for name, param in self.diffusion.named_parameters():
                if "lora_" in name:
                    lora_state_dict[name] = param.cpu().clone()
            return lora_state_dict

    def load_lora_weights(self, lora_state_dict: dict, strict: bool = True):
        """Load LoRA weights into the model."""
        missing_keys, unexpected_keys = [], []
        for name, param in lora_state_dict.items():
            if name in dict(self.diffusion.named_parameters()):
                dict(self.diffusion.named_parameters())[name].data.copy_(param)
            else:
                unexpected_keys.append(name)

        if strict:
            for name, param in self.diffusion.named_parameters():
                if "lora_" in name and name not in lora_state_dict:
                    missing_keys.append(name)

            if missing_keys or unexpected_keys:
                error_msg = f"Error loading LoRA weights:\n"
                if missing_keys:
                    error_msg += f"Missing keys: {missing_keys}\n"
                if unexpected_keys:
                    error_msg += f"Unexpected keys: {unexpected_keys}\n"
                raise RuntimeError(error_msg)

    def merge_lora_weights(self):
        """Merge LoRA weights into the base model and remove LoRA adapters.

        This is useful when you want to:
        1. Create a standalone model with LoRA weights baked in
        2. Train a new LoRA on top of the merged weights

        After calling this method, the model will no longer have LoRA adapters,
        and all LoRA weights will be merged into the base model parameters.

        Returns:
            None
        """
        if not hasattr(self.diffusion, "merge_and_unload"):
            raise RuntimeError("Model does not have LoRA adapters. Cannot merge.")

        # `merge_and_unload()` returns a new model with LoRA weights merged
        self.diffusion = self.diffusion.merge_and_unload()
