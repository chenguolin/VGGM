from typing import *
from torch import Tensor

import os
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as tF
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from einops import rearrange
from pytorch_msssim import ssim as SSIM
from lpips import LPIPS

from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options
from src.models.networks import (
    FeatureEmbed,
    WanTextEncoderWrapper,
    WanVAEWrapper,
    WanDiffusionWrapper,
    WanDiffusionDA3Wrapper,
)
from src.models.losses import XYZLoss, DepthLoss, CameraLoss
from src.utils import convert_to_buffer, plucker_ray, zero_init_module, colorize_depth


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Wan(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        # Text encoder
        if opt.load_text_encoder:
            self.text_encoder = WanTextEncoderWrapper(opt.wan_dir)
            if opt.use_deepspeed_zero3:
                self.text_encoder.requires_grad_(False)  # for ZeRO3 parameter split
            else:
                convert_to_buffer(self.text_encoder, persistent=False)  # no gradient & not save to checkpoint
        else:
            self.text_encoder = None

        # Diffusion model
        if not opt.load_da3:
            self.diffusion = WanDiffusionWrapper(
                opt.wan_dir,
                opt.num_train_timesteps,
                opt.num_inference_steps,
                opt.shift,
                opt.sigma_min,
                opt.extra_one_step,
                #
                opt.use_gradient_checkpointing,
                opt.use_gradient_checkpointing_offload,
                #
                is_causal=opt.is_causal,
                sink_size=opt.sink_size,
                chunk_size=opt.chunk_size,
                max_attention_size=opt.max_attention_size,
                rope_outside=opt.rope_outside,
            )
        else:
            self.diffusion = WanDiffusionDA3Wrapper(
                opt.wan_dir,
                opt.num_train_timesteps,
                opt.num_inference_steps,
                opt.shift,
                opt.sigma_min,
                opt.extra_one_step,
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
                da3_use_ray_pose=opt.da3_use_ray_pose,
                da3_use_bicrossattn=opt.da3_use_bicrossattn and not opt.only_train_da3,
                da3_max_attention_size=opt.da3_max_attention_size,
            )
            if opt.only_train_da3:
                self.diffusion.requires_grad_(False)
                self.diffusion.da3_adapter.requires_grad_(True)
                self.diffusion.da3_model.requires_grad_(True)

            self.ray_loss_fn, self.depth_loss_fn, self.pose_loss_fn = \
                XYZLoss(opt), DepthLoss(opt), CameraLoss(opt)

        if opt.generator_path is not None:
            state_dict = torch.load(opt.generator_path, map_location="cpu", weights_only=True)
            if "generator_ema" in state_dict:
                self.diffusion.load_state_dict(state_dict["generator_ema"], strict=False)
            elif "generator" in state_dict:
                self.diffusion.load_state_dict(state_dict["generator"], strict=False)
            else:
                self.diffusion.load_state_dict(state_dict, strict=False)

        # (Optional) Plucker embeddings
        if opt.input_plucker:
            self.plucker_embed = FeatureEmbed(
                "causal3d",
                input_channels=6,
                out_channels=self.diffusion.model.dim,
                t_ratio=opt.compression_ratio[0],
                s_ratio=opt.compression_ratio[1] * self.diffusion.model.patch_size[1],
            )
            zero_init_module(self.plucker_embed)
            if opt.plucker_embed_path is not None:
                self.plucker_embed.load_state_dict(torch.load(opt.plucker_embed_path, map_location="cpu", weights_only=True))
                convert_to_buffer(self.plucker_embed, persistent=False)  # no gradient & not save to checkpoint

        # Add LoRA in the diffusion model, will freeze all parameters except LoRA layers
        if opt.use_lora_in_wan:
            self._add_lora_to_wan(
                target_modules=opt.lora_target_modules_in_wan.split(","),
                lora_rank=opt.lora_rank_in_wan,
            )

        # Set other trainable parameters except LoRA layers in the diffusion model
        if opt.more_trainable_wan_params is not None:
            trainble_names = opt.more_trainable_wan_params.split(",")
            if opt.use_lora_in_wan:
                trainble_names.append("lora")
            for name, param in self.diffusion.model.named_parameters():
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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        if self.text_encoder is not None and "text_encoder" in state_dict:
            del state_dict["text_encoder"]
        return state_dict

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Any], dtype: torch.dtype = torch.float32, is_eval: bool = False, vae: Optional[WanVAEWrapper] = None):
        outputs = {}

        if "image" in data:
            images = data["image"].to(dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W), device = images.shape, images.device
        else:
            B = len(data["prompt"])
            F, H, W = self.opt.num_input_frames, self.opt.input_res[0], self.opt.input_res[1]
            device = self.diffusion.model.device

        if self.opt.load_da3:
            depths = data["depth"].to(dtype)  # (B, F, H, W)
        C2W = data["C2W"].to(dtype)  # (B, F, 4, 4)
        fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, F, 4)

        # Text encoder
        if self.text_encoder is not None:
            prompts = data["prompt"]  # a list of strings
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                self.text_encoder.eval()
                prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
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

        # (Optional) Plucker embeddings
        if self.opt.input_plucker:
            plucker, _ = plucker_ray(H, W, C2W.float(), fxfycxcy.float())  # (B, F, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w").to(dtype)
            plucker_embeds = self.plucker_embed(plucker)  # (B, D, f, hh, ww)
        else:
            plucker_embeds = None

        # Diffusion
        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)
        noises = torch.randn_like(latents)

        min_t, max_t = int(self.opt.min_timestep_boundary * self.opt.num_train_timesteps), \
            int(self.opt.max_timestep_boundary * self.opt.num_train_timesteps)
        if not self.opt.is_causal:
            timesteps_id = torch.randint(min_t, max_t, (1,))  # (1,); batch share the same timestep for simpler time scheduler
            timesteps_id = timesteps_id.unsqueeze(1).repeat(B, f)  # (B, f)
        else:  # teacher / diffusion forcing
            assert f % self.opt.chunk_size == 0
            num_chunks = f // self.opt.chunk_size

            timesteps_id = torch.randint(min_t, max_t, (num_chunks,))  # (num_chunks,); each chunk in different noise level
            timesteps_id = timesteps_id.repeat_interleave(self.opt.chunk_size, dim=0).repeat(B, 1)  # (B, f); batch share the same timestep for simpler time scheduler
        timesteps = self.diffusion.scheduler.timesteps[timesteps_id].to(dtype=dtype, device=device)

        noisy_latents = self.diffusion.scheduler.add_noise(
            latents.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
            noises.transpose(1, 2).flatten(0, 1),   # (B*f, D, h, w)
            timesteps.flatten(0, 1),                # (B*f,)
        ).detach().unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)
        targets = self.diffusion.scheduler.training_target(latents, noises)

        # # Classifier-free guidance dropout
        # if self.training:
        #     masks = (torch.rand(B, device=device) < self.opt.cfg_dropout).to(dtype)
        #     # prompt_embeds = prompt_embeds * masks[:, None, None]# + negative_prompt_embeds * (1 - masks)[:, None, None]
        #     # if plucker_embeds is not None:
        #     #     plucker_embeds = plucker_embeds * masks[:, None, None, None, None]

        if cond_latents is not None:
            model_outputs = self.diffusion(
                torch.cat([cond_latents, noisy_latents[:, :, 1:, ...]], dim=2),
                torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1),
                prompt_embeds,
                add_embeds=plucker_embeds,
                #
                clean_x=latents if self.opt.use_teacher_forcing else None,
            )
        else:
            model_outputs = self.diffusion(
                noisy_latents,
                timesteps,
                prompt_embeds,
                add_embeds=plucker_embeds,
                #
                clean_x=latents if self.opt.use_teacher_forcing else None,
            )

        if self.opt.load_da3:
            model_outputs, da3_outputs = model_outputs

        # Diffusion loss
        diffusion_loss = tF.mse_loss(model_outputs.float(), targets.float(), reduction="none")  # (B, D, f, h, w)
        diffusion_loss = self.diffusion.scheduler.training_weight(timesteps.flatten(0, 1)).reshape(-1, 1, 1, 1) * \
            diffusion_loss.transpose(1, 2).flatten(0, 1)  # (B*f, D, h, w)
        diffusion_loss = diffusion_loss.unflatten(0, (B, f)).transpose(1, 2)  # (B, D, f, h, w)
        if cond_latents is not None:
            outputs["diffusion_loss"] = diffusion_loss[:, :, 1:, ...].mean()
        else:
            outputs["diffusion_loss"] = diffusion_loss.mean()
        outputs["loss"] = outputs["diffusion_loss"]

        # (Optional) DA3 loss
        if self.opt.load_da3:
            ## Get ground-truth geometry labels
            idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
            gt_depths = depths[:, idxs, ...]  # (B, f, H, W)
            _, (ray_o, ray_d) = plucker_ray(H//2, W//2,
                C2W[:, idxs, ...].float(), fxfycxcy[:, idxs, ...].float(), normalize_ray_d=False)
            gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
            gt_pose_enc = torch.cat([
                C2W[:, idxs, :3, 3].float(),  # (B, f, 3)
                mat_to_quat(C2W[:, idxs, :3, :3].float()),  # (B, f, 4)
                2. * torch.atan(1. / (2. * fxfycxcy[:, idxs, 1:2])),  # (B, f, 1); fy -> fov_h
                2. * torch.atan(1. / (2. * fxfycxcy[:, idxs, 0:1])),  # (B, f, 1); fx -> fov_w
            ], dim=-1).to(dtype)  # (B, f, 9)
            if self.opt.use_teacher_forcing:
                gt_depths, gt_raymaps, gt_pose_enc = \
                    torch.cat([gt_depths] * 2, dim=1), torch.cat([gt_raymaps] * 2, dim=1), torch.cat([gt_pose_enc] * 2, dim=1)
            ## Compute geometry losses
            outputs["depth_loss"] = self.depth_loss_fn(da3_outputs["depth"], gt_depths, confs=da3_outputs["depth_conf"])  # (,)
            outputs["ray_loss"] = self.ray_loss_fn(da3_outputs["ray"], gt_raymaps, confs=da3_outputs["ray_conf"])  # (,)
            outputs["pose_loss"] = self.pose_loss_fn(da3_outputs["pose_enc"], gt_pose_enc)  # (,)
            outputs["loss"] = outputs["diffusion_loss"] + \
                outputs["depth_loss"] + outputs["ray_loss"] + outputs["pose_loss"]

        # For visualizaiton
        if is_eval:
            if cond_latents is not None:
                pred_x0 = self.diffusion._convert_flow_pred_to_x0(
                    model_outputs,
                    torch.cat([cond_latents, noisy_latents[:, :, 1:, ...]], dim=2),
                    torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1),
                ).to(dtype)
            else:
                pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_latents, timesteps).to(dtype)

            outputs["images_predx0"] = (self.decode_latent(pred_x0, vae).clamp(-1., 1.) + 1.) / 2.
            if "image" in data:
                outputs["images_recon"] = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.

            if self.opt.load_da3:
                outputs["images_gt_depth"] = colorize_depth(1./gt_depths, batch_mode=True)
                outputs["images_pred_depth"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)

        return outputs

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def evaluate(self, data: Dict[str, Any], dtype: torch.dtype = torch.bfloat16, verbose: bool = True, vae: Optional[WanVAEWrapper] = None):
        verbose = verbose and (not dist.is_initialized() or dist.get_rank() == 0)
        return self.evaluate_causal(data, dtype, verbose, vae) if self.opt.is_causal else self.evaluate_bidirectional(data, dtype, verbose, vae)

    def evaluate_bidirectional(self, data: Dict[str, Any], dtype: torch.dtype, verbose: bool = True, vae: Optional[WanVAEWrapper] = None):
        outputs = {}

        self.diffusion.eval()
        self.diffusion.scheduler.set_timesteps(self.opt.num_inference_steps, training=False)

        if "image" in data:
            images = data["image"].to(dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W), device = images.shape, images.device

            # For visualization
            outputs["images_gt"] = images
        else:
            B = len(data["prompt"])
            F, H, W = self.opt.num_input_frames_test, self.opt.input_res[0], self.opt.input_res[1]
            device = self.diffusion.model.device

        if self.opt.load_da3:
            depths = data["depth"].to(dtype)  # (B, F, H, W)
        C2W = data["C2W"].to(dtype)  # (B, F, 4, 4)
        fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, F, 4)

        f = 1 + (F - 1) // self.opt.compression_ratio[0]
        h = H // self.opt.compression_ratio[1]
        w = W // self.opt.compression_ratio[2]

        # Text encoder
        if self.text_encoder is not None:
            prompts = data["prompt"]  # a list of strings
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
            negative_prompt_embeds = self.text_encoder([self.opt.negative_prompt]).repeat(B, 1, 1)  # (B, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if self.opt.first_latent_cond and "image" in data:
            cond_latents = self.encode(images[:, 0:1, ...] * 2. - 1., vae)  # (B, D, 1, h, w)
        else:
            cond_latents = None
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Plucker embeddings
        if self.opt.input_plucker:
            plucker, _ = plucker_ray(H, W, C2W.float(), fxfycxcy.float())  # (B, F, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w").to(dtype)
            plucker_embeds = self.plucker_embed(plucker)  # (B, D, f, hh, ww)
        else:
            plucker_embeds = None

        # Denoising
        for cfg_scale in self.opt.cfg_scale:
            latents = torch.randn(B, self.opt.latent_dim, f, h, w, device=device, dtype=dtype)

            for i, timestep in tqdm(enumerate(self.diffusion.scheduler.timesteps),
                total=len(self.diffusion.scheduler.timesteps), ncols=125, disable=not verbose, desc="[Denoise]"):
                timesteps = timestep * torch.ones(B, f).to(dtype=dtype, device=device)

                ## Diffusion model
                if cond_latents is not None:
                    model_outputs = self.diffusion(
                        torch.cat([cond_latents, latents[:, :, 1:, ...]], dim=2),
                        torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1),
                        prompt_embeds,
                        add_embeds=plucker_embeds,
                    )
                else:
                    model_outputs = self.diffusion(
                        latents,
                        timesteps,
                        prompt_embeds,
                        add_embeds=plucker_embeds,
                    )

                ## CFG
                if cfg_scale > 1.:
                    if cond_latents is not None:
                        model_outputs_neg = self.diffusion(
                            torch.cat([cond_latents, latents[:, :, 1:, ...]], dim=2),
                            torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1),
                            negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                            add_embeds=plucker_embeds,  # torch.zeros_like(plucker_embeds) if plucker_embeds is not None else None,
                        )
                    else:
                        model_outputs_neg = self.diffusion(
                            latents,
                            timesteps,
                            negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                            add_embeds=plucker_embeds,  # torch.zeros_like(plucker_embeds) if plucker_embeds is not None else None,
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
                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, latents,
                        timestep * torch.ones_like(timesteps))
                    if i < len(self.diffusion.scheduler.timesteps) - 1:
                        latents = self.diffusion.scheduler.add_noise(
                            pred_x0.transpose(1, 2).flatten(0, 1),
                            torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                            self.diffusion.scheduler.timesteps[i + 1] * torch.ones_like(timesteps).flatten(0, 1),
                        ).unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)
                    else:
                        latents = pred_x0

            if cond_latents is not None:
                latents[:, :, 0:1, ...] = cond_latents

            # Decode
            pred_images = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.  # (B, D, f, h, w) -> (B, F, 3, H, W)
            outputs[f"images_pred_{cfg_scale}"] = pred_images

            # Evaluation metrics: PSNR, SSIM, LPIPS
            if "image" in data:
                outputs[f"psnr_{cfg_scale}"] = -10. * torch.log10(torch.mean((images - pred_images) ** 2, dim=(1, 2, 3, 4)))  # (B,)
                outputs[f"ssim_{cfg_scale}"] = SSIM(
                    rearrange(pred_images, "b f c h w -> (b f) c h w"),
                    rearrange(images, "b f c h w -> (b f) c h w"),
                    data_range=1., size_average=False,
                )  # (B*F,)
                outputs[f"ssim_{cfg_scale}"] = rearrange(outputs[f"ssim_{cfg_scale}"], "(b f) -> b f", b=B).mean(dim=1)  # (B,)
                if self.lpips_loss is not None:
                    outputs[f"lpips_{cfg_scale}"] = self.lpips_loss(
                        rearrange(pred_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                        rearrange(images, "b f c h w -> (b f) c h w") * 2. - 1.,
                    )  # (B*F, 1, 1, 1)
                    outputs[f"lpips_{cfg_scale}"] = rearrange(outputs[f"lpips_{cfg_scale}"], "(b f) c h w -> b f c h w", b=B).mean(dim=(1, 2, 3, 4))  # (B,)

            # (Optional) DA3 evaluation
            if self.opt.load_da3:
                assert da3_outputs is not None

                ## Get ground-truth geometry labels
                idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
                gt_depths = depths[:, idxs, ...]  # (B, f, H, W)
                _, (ray_o, ray_d) = plucker_ray(H//2, W//2,
                    C2W[:, idxs, ...].float(), fxfycxcy[:, idxs, ...].float(), normalize_ray_d=False)
                gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
                gt_pose_enc = torch.cat([
                    C2W[:, idxs, :3, 3].float(),  # (B, f, 3)
                    mat_to_quat(C2W[:, idxs, :3, :3].float()),  # (B, f, 4)
                    2. * torch.atan(1. / (2. * fxfycxcy[:, idxs, 1:2])),  # (B, f, 1); fy -> fov_h
                    2. * torch.atan(1. / (2. * fxfycxcy[:, idxs, 0:1])),  # (B, f, 1); fx -> fov_w
                ], dim=-1).to(dtype)  # (B, f, 9)
                outputs[f"images_gt_depth"] = colorize_depth(1./gt_depths, batch_mode=True)

                ## Compute geometry metrics via MSE
                outputs[f"depth_{cfg_scale}"] = tF.mse_loss(da3_outputs["depth"], gt_depths)  # (,)
                outputs[f"ray_{cfg_scale}"] = tF.mse_loss(da3_outputs["ray"], gt_raymaps)  # (,)
                outputs[f"pose_{cfg_scale}"] = tF.mse_loss(da3_outputs["pose_enc"], gt_pose_enc)  # (,)

                # For visualization
                outputs[f"images_pred_depth_{cfg_scale}"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)

        return outputs

    def evaluate_causal(self, data: Dict[str, Any], dtype: torch.dtype, verbose: bool = True, vae: Optional[WanVAEWrapper] = None):
        outputs = {}

        self.diffusion.eval()
        self.diffusion.scheduler.set_timesteps(self.opt.num_inference_steps, training=False)

        if "image" in data:
            images = data["image"].to(dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W), device = images.shape, images.device

            # For visualization
            outputs["images_gt"] = images
        else:
            B = len(data["prompt"])
            F, H, W = self.opt.num_input_frames_test, self.opt.input_res[0], self.opt.input_res[1]
            device = self.diffusion.model.device
            images = torch.zeros((B, F, 3, H, W), dtype=dtype, device=device)  # (B, F, 3, H, W); not really used

        if self.opt.load_da3:
            depths = data["depth"].to(dtype)  # (B, F, H, W)
        C2W = data["C2W"].to(dtype)  # (B, F, 4, 4)
        fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, F, 4)

        f = 1 + (F - 1) // self.opt.compression_ratio[0]
        h = H // self.opt.compression_ratio[1]
        w = W // self.opt.compression_ratio[2]

        # Text encoder
        if self.text_encoder is not None:
            prompts = data["prompt"]  # a list of strings
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
            negative_prompt_embeds = self.text_encoder([self.opt.negative_prompt]).repeat(B, 1, 1)  # (B, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if self.opt.first_latent_cond and "image" in data:
            cond_latents = self.encode(images[:, 0:1, ...] * 2. - 1., vae)  # (B, D, 1, h, w)
        else:
            cond_latents = None
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Plucker embeddings
        if self.opt.input_plucker:
            plucker, _ = plucker_ray(H, W, C2W.float(), fxfycxcy.float())  # (B, F, 6, H, W)
            plucker = rearrange(plucker, "b f c h w -> b c f h w").to(dtype)
            plucker_embeds = self.plucker_embed(plucker)  # (B, D, f, hh, ww)
        else:
            plucker_embeds = None

        # Denoising
        for cfg_scale in self.opt.cfg_scale:
            latents = torch.randn(B, self.opt.latent_dim, f, h, w, device=device, dtype=dtype)

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
                this_chunk_plucker_embeds = None
                if plucker_embeds is not None:
                    this_chunk_plucker_embeds = plucker_embeds[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]

                ### Spatial denoising loop
                for i, timestep in tqdm(enumerate(self.diffusion.scheduler.timesteps),
                    total=len(self.diffusion.scheduler.timesteps), ncols=125, disable=not verbose, desc="[Denoise]"):
                    timesteps = timestep[None, None].repeat(B, self.opt.chunk_size).to(dtype=dtype, device=device)

                    #### Diffusion model
                    if chunk_idx == 0 and cond_latents is not None:
                        model_outputs = self.diffusion(
                            torch.cat([cond_latents, this_chunk_latents[:, :, 1:, ...]], dim=2),
                            torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1),
                            prompt_embeds,
                            add_embeds=this_chunk_plucker_embeds,
                            #
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                            #
                            kv_cache_da3=self.kv_cache_pos_da3,
                            current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen + 1),  # `+1` for camera token
                        )
                    else:
                        model_outputs = self.diffusion(
                            this_chunk_latents,
                            timesteps,
                            prompt_embeds,
                            add_embeds=this_chunk_plucker_embeds,
                            #
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                            #
                            kv_cache_da3=self.kv_cache_pos_da3,
                            current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen + 1),  # `+1` for camera token
                        )

                    #### CFG
                    if cfg_scale > 1.:
                        if chunk_idx == 0 and cond_latents is not None:
                            model_outputs_neg = self.diffusion(
                                torch.cat([cond_latents, this_chunk_latents[:, :, 1:, ...]], dim=2),
                                torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1),
                                negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                                add_embeds=this_chunk_plucker_embeds,  # torch.zeros_like(this_chunk_plucker_embeds) if this_chunk_plucker_embeds is not None else None,
                                #
                                kv_cache=self.kv_cache_neg,
                                crossattn_cache=self.crossattn_cache_neg,
                                current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                                #
                                kv_cache_da3=self.kv_cache_neg_da3,
                                current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen + 1),  # `+1` for camera token
                            )
                        else:
                            model_outputs_neg = self.diffusion(
                                this_chunk_latents,
                                timesteps,
                                negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                                add_embeds=this_chunk_plucker_embeds,  # torch.zeros_like(this_chunk_plucker_embeds) if this_chunk_plucker_embeds is not None else None,
                                #
                                kv_cache=self.kv_cache_neg,
                                crossattn_cache=self.crossattn_cache_neg,
                                current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                                #
                                kv_cache_da3=self.kv_cache_neg_da3,
                                current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen + 1),  # `+1` for camera token
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
                            timesteps.flatten(0, 1),
                            this_chunk_latents.transpose(1, 2).flatten(0, 1),
                        ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
                    else:
                        pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents,
                            timestep * torch.ones_like(timesteps))
                        if i < len(self.diffusion.scheduler.timesteps) - 1:
                            this_chunk_latents = self.diffusion.scheduler.add_noise(
                                pred_x0.transpose(1, 2).flatten(0, 1),
                                torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                                self.diffusion.scheduler.timesteps[i + 1] * torch.ones_like(timesteps).flatten(0, 1),
                            ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)  # (B, D, f_chunk, h, w)
                        else:
                            this_chunk_latents = pred_x0

                if chunk_idx == 0 and cond_latents is not None:
                    this_chunk_latents[:, :, 0:1, ...] = cond_latents

                # Record this chunk generated latents
                latents[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...] = this_chunk_latents

                # Rerun with timestep zero to update KV cache
                # TODO: add noise on KV cache, except the first chunk
                if self.opt.extra_one_step and chunk_idx < num_chunks - 1:
                    model_outputs = self.diffusion(
                        this_chunk_latents,
                        timesteps * 0.,
                        prompt_embeds,
                        add_embeds=this_chunk_plucker_embeds,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen + 1),  # `+1` for camera token
                    )
                    if cfg_scale > 1.:
                        model_outputs_neg = self.diffusion(
                            this_chunk_latents,
                            timesteps * 0.,
                            negative_prompt_embeds,  # torch.zeros_like(prompt_embeds)
                            add_embeds=this_chunk_plucker_embeds,  # torch.zeros_like(this_chunk_plucker_embeds) if this_chunk_plucker_embeds is not None else None,
                            #
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                            #
                            kv_cache_da3=self.kv_cache_neg_da3,
                            current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen + 1),  # `+1` for camera token
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

            if cond_latents is not None:
                latents[:, :, 0:1, ...] = cond_latents

            # Decode
            pred_images = (self.decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.  # (B, D, f, h, w) -> (B, F, 3, H, W)
            outputs[f"images_pred_{cfg_scale}"] = pred_images

            # Evaluation metrics: PSNR, SSIM, LPIPS
            if "image" in data:
                outputs[f"psnr_{cfg_scale}"] = -10. * torch.log10(torch.mean((images - pred_images) ** 2, dim=(1, 2, 3, 4)))  # (B,)
                outputs[f"ssim_{cfg_scale}"] = SSIM(
                    rearrange(pred_images, "b f c h w -> (b f) c h w"),
                    rearrange(images, "b f c h w -> (b f) c h w"),
                    data_range=1., size_average=False,
                )  # (B*F,)
                outputs[f"ssim_{cfg_scale}"] = rearrange(outputs[f"ssim_{cfg_scale}"], "(b f) -> b f", b=B).mean(dim=1)  # (B,)
                if self.lpips_loss is not None:
                    outputs[f"lpips_{cfg_scale}"] = self.lpips_loss(
                        rearrange(pred_images, "b f c h w -> (b f) c h w") * 2. - 1.,
                        rearrange(images, "b f c h w -> (b f) c h w") * 2. - 1.,
                    )  # (B*F, 1, 1, 1)
                    outputs[f"lpips_{cfg_scale}"] = rearrange(outputs[f"lpips_{cfg_scale}"], "(b f) c h w -> b f c h w", b=B).mean(dim=(1, 2, 3, 4))  # (B,)

            # (Optional) DA3 evaluation
            if self.opt.load_da3:
                assert da3_outputs is not None
                da3_outputs = {
                    k: torch.cat([all_da3_outputs[i][k] for i in range(num_chunks)], dim=1)
                    for k in all_da3_outputs[0].keys()
                }

                ## Get ground-truth geometry labels
                idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
                gt_depths = depths[:, idxs, ...]  # (B, f, H, W)
                _, (ray_o, ray_d) = plucker_ray(H//2, W//2,
                    C2W[:, idxs, ...].float(), fxfycxcy[:, idxs, ...].float(), normalize_ray_d=False)
                gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
                gt_pose_enc = torch.cat([
                    C2W[:, idxs, :3, 3].float(),  # (B, f, 3)
                    mat_to_quat(C2W[:, idxs, :3, :3].float()),  # (B, f, 4)
                    2. * torch.atan(1. / (2. * fxfycxcy[:, idxs, 1:2])),  # (B, f, 1); fy -> fov_h
                    2. * torch.atan(1. / (2. * fxfycxcy[:, idxs, 0:1])),  # (B, f, 1); fx -> fov_w
                ], dim=-1).to(dtype)  # (B, f, 9)
                outputs[f"images_gt_depth"] = colorize_depth(1./gt_depths, batch_mode=True)

                ## Compute geometry metrics via MSE
                outputs[f"depth_{cfg_scale}"] = tF.mse_loss(da3_outputs["depth"], gt_depths)  # (,)
                outputs[f"ray_{cfg_scale}"] = tF.mse_loss(da3_outputs["ray"], gt_raymaps)  # (,)
                outputs[f"pose_{cfg_scale}"] = tF.mse_loss(da3_outputs["pose_enc"], gt_pose_enc)  # (,)

                # For visualization
                outputs[f"images_pred_depth_{cfg_scale}"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)

        return outputs


    ################################ Helper functions ################################


    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """
        Initialize a per-GPU KV cache for the Wan model.
        """
        num_blocks = len(self.diffusion.model.blocks)
        num_heads = self.diffusion.model.num_heads
        head_dim = self.diffusion.model.dim // num_heads

        kv_cache_pos, kv_cache_neg = [], []
        for _ in range(num_blocks):
            kv_cache_pos.append({
                "k": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads, head_dim), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
            kv_cache_neg.append({
                "k": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads, head_dim), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

        if self.opt.load_da3:
            num_da3_blocks = len(self.diffusion.da3_model.backbone.pretrained.blocks)
            num_heads_da3 = self.diffusion.da3_model.backbone.pretrained.num_heads
            head_dim_da3 = self.diffusion.da3_model.backbone.pretrained.embed_dim // num_heads_da3

            kv_cache_pos_da3, kv_cache_neg_da3 = [], []
            for _ in range(num_da3_blocks):
                kv_cache_pos_da3.append({
                    "k": torch.zeros((batch_size, num_heads_da3, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, num_heads_da3, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                })
                kv_cache_neg_da3.append({
                    "k": torch.zeros((batch_size, num_heads_da3, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, num_heads_da3, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
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
        self.diffusion.model = inject_adapter_in_model(lora_config, self.diffusion.model)
