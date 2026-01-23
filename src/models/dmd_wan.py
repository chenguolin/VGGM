from typing import *
from torch import Tensor

import os
import numpy as np
import torch
import torch.nn.functional as tF
from peft import LoraConfig, inject_adapter_in_model

from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options
from src.models.networks import WanDiffusionWrapper, WanVAEWrapper
from src.models.wan import Wan
from src.models.pipelines.self_forcing_training import SelfForcingTrainingPipeline
from src.utils import convert_to_buffer, plucker_ray, colorize_depth, filter_da3_points, render_pt3d_points


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DMD_Wan(Wan):
    def __init__(self, opt: Options):
        super().__init__(opt)

        # Real score model for DMD
        self.real_score = WanDiffusionWrapper(
            opt.wan_dir,
            opt.num_train_timesteps,
            opt.num_inference_steps,
            opt.shift,
            0.,    # hard-coded `sigma_min`
            True,  # hard-coded `extra_one_step`
            #
            opt.teacher_input_plucker,
            opt.teacher_extra_condition_dim,
            #
            0,  # hard-coded `memory_num_tokens`
            #
            opt.use_gradient_checkpointing,
            opt.use_gradient_checkpointing_offload,
            #
            is_causal=opt.is_teacher_causal,
            sink_size=opt.sink_size,
            chunk_size=opt.chunk_size,
            max_attention_size=opt.max_attention_size,
        )
        if opt.teacher_path is not None:
            state_dict = torch.load(opt.teacher_path, map_location="cpu", weights_only=True)
            if "generator_ema" in state_dict:
                self.real_score.load_state_dict(state_dict["generator_ema"])
            elif "generator" in state_dict:
                self.real_score.load_state_dict(state_dict["generator"])
            else:
                self.real_score.load_state_dict(state_dict)
        if opt.use_deepspeed_zero3:
            self.real_score.requires_grad_(False)  # for ZeRO3 parameter split
        else:
            convert_to_buffer(self.real_score, persistent=False)  # no gradient & not save to checkpoint

        # Fake score model for DMD
        self.fake_score = WanDiffusionWrapper(
            opt.wan_dir,
            opt.num_train_timesteps,
            opt.num_inference_steps,
            opt.shift,
            0.,    # hard-coded `sigma_min`
            True,  # hard-coded `extra_one_step`
            #
            opt.teacher_input_plucker,
            opt.teacher_extra_condition_dim,
            #
            0,  # hard-coded `memory_num_tokens`
            #
            opt.use_gradient_checkpointing,
            opt.use_gradient_checkpointing_offload,
            #
            is_causal=opt.is_teacher_causal,
            sink_size=opt.sink_size,
            chunk_size=opt.chunk_size,
            max_attention_size=opt.max_attention_size,
        )
        if opt.teacher_path is not None:
            state_dict = torch.load(opt.teacher_path, map_location="cpu", weights_only=True)
            if "generator_ema" in state_dict:
                self.fake_score.load_state_dict(state_dict["generator_ema"])
            elif "generator" in state_dict:
                self.fake_score.load_state_dict(state_dict["generator"])
            else:
                self.fake_score.load_state_dict(state_dict)

        # This will be init later with fsdp/deepspeed-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Add LoRA in the diffusion model, will freeze all parameters except LoRA layers
        if opt.use_lora_in_fake_score:
            self._add_lora_to_fake_score(
                target_modules=opt.lora_target_modules_in_fake_score.split(","),
                lora_rank=opt.lora_rank_in_fake_score,
            )

        # Set other trainable parameters except LoRA layers in the diffusion model
        if opt.more_trainable_fake_score_params is not None:
            trainble_names = opt.more_trainable_fake_score_params.split(",")
            if opt.use_lora_in_fake_score:
                trainble_names.append("lora")
            for name, param in self.fake_score.model.named_parameters():
                _flag = False
                for trainble_name in trainble_names:
                    if trainble_name in name:
                        param.requires_grad_(True)
                        _flag = True
                        break
                if not _flag:
                    param.requires_grad_(False)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        if self.text_encoder is not None and "text_encoder" in state_dict:
            del state_dict["text_encoder"]
        if self.real_score is not None and "real_score" in state_dict:
            del state_dict["real_score"]
        return state_dict

    def compute_loss(self, data: Dict[str, Any], dtype: torch.dtype = torch.float32, train_generator: bool = True, is_eval: bool = False, vae: Optional[WanVAEWrapper] = None):
        outputs = {}

        # CausVid or Self-Forcing
        self.use_self_forcing = use_self_forcing = np.random.rand() <= self.opt.self_forcing_prob

        if "image" in data and (not use_self_forcing or self.opt.first_latent_cond or self.opt.teacher_first_latent_cond):
            images = data["image"].to(dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W), device = images.shape, images.device
        else:
            B = len(data["prompt"])
            F, H, W = self.opt.num_input_frames, self.opt.input_res[0], self.opt.input_res[1]
            device = self.diffusion.model.device

        # Text encoder
        if self.text_encoder is not None:
            if self.prompt_list is None or not self.use_self_forcing:
                prompts = data["prompt"]  # a list of strings
            else:
                prompts = np.random.choice(self.prompt_list, B, replace=False).tolist()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                self.text_encoder.eval()
                prompt_embeds = self.text_encoder(prompts)  # (B, N=512, D')
                negative_prompt_embeds = self.text_encoder([self.opt.negative_prompt]).repeat(B, 1, 1)  # (B, N=512, D')
        else:
            raise NotImplementedError

        # VAE
        if "image" in data and self.prompt_list is None and (not use_self_forcing or self.opt.first_latent_cond or self.opt.teacher_first_latent_cond):
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
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Camera & depth
        idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
        if "C2W" in data and "fxfycxcy" in data:
            C2W = data["C2W"].to(dtype)[:, idxs, ...]  # (B, f, 4, 4)
            fxfycxcy = data["fxfycxcy"].to(dtype)[:, idxs, ...]  # (B, f, 4)
            plucker = plucker_ray(H, W, C2W.float(), fxfycxcy.float())[0].to(dtype)  # (B, f, 6, H, W)
        else:
            C2W, fxfycxcy, plucker = None, None, None
        if "depth" in data:
            depths = data["depth"].to(dtype)[:, idxs, ...]  # (B, f, H, W)
        else:
            depths = None
        if "conf" in data:
            confs = data["conf"].to(dtype)[:, idxs, ...]  # (B, f, H, W)
        else:
            confs = None
        if "image" in data:
            images_f = data["image"].to(dtype)[:, idxs, ...]  # (B, f, 3, H, W)
        else:
            images_f = None

        # DMD
        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)

        if train_generator:
            generator_loss, dmd_grad_norm, pred_x0, da3_outputs = \
                self.generator_loss(
                    torch.randn_like(latents),
                    prompt_embeds,
                    negative_prompt_embeds,
                    cond_latents,
                    plucker,
                    #
                    clean_latents=latents if not use_self_forcing else None,
                    #
                    C2W=C2W,
                    fxfycxcy=fxfycxcy,
                    depths=depths,
                    confs=confs,
                    images_f=images_f,
                )
            outputs["generator_loss"] = generator_loss
            outputs["dmd_grad_norm"] = dmd_grad_norm

            if da3_outputs is not None:
                if "depth_loss" in da3_outputs:
                    outputs["depth_loss"] = da3_outputs["depth_loss"]
                    generator_loss = generator_loss + da3_outputs["depth_loss"]
                if "ray_loss" in da3_outputs:
                    outputs["ray_loss"] = da3_outputs["ray_loss"]
                    generator_loss = generator_loss + da3_outputs["ray_loss"]
                if "camera_loss" in da3_outputs:
                    outputs["camera_loss"] = da3_outputs["camera_loss"]
                    generator_loss = generator_loss + da3_outputs["camera_loss"]
                if "render_loss" in da3_outputs:
                    outputs["render_loss"] = da3_outputs["render_loss"]
                    generator_loss = generator_loss + da3_outputs["render_loss"]

        else:
            generator_loss = 0.

        outputs["critic_loss"] = critic_loss = \
            self.critic_loss(
                torch.randn_like(latents),
                prompt_embeds,
                cond_latents,
                plucker,
                #
                clean_latents=latents if not use_self_forcing else None,
                #
                C2W=C2W,
                fxfycxcy=fxfycxcy,
                depths=depths,
                confs=confs,
                images_f=images_f,
            )

        outputs["loss"] = critic_loss + self.opt.dmd_loss_weight * generator_loss

        # For visualizaiton
        if is_eval:
            outputs["images_predx0"] = (self.decode_latent(pred_x0, vae).clamp(-1., 1.) + 1.) / 2.
            if "image" in data:
                outputs["images_input"] = data["image"]
            if da3_outputs is not None:
                if "depth" in da3_outputs:
                    outputs["images_pred_depth"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)
                if "images_render" in da3_outputs:
                    outputs["images_render"] = da3_outputs["images_render"]
                if "images_render_depth" in da3_outputs:
                    outputs["images_render_depth"] = da3_outputs["images_render_depth"]

        return outputs

    ################################ Helper functions ################################

    def critic_loss(self,
        noises: Tensor,
        prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        clean_latents: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
        depths: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
        images_f: Optional[Tensor] = None,
    ):
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        """
        B, f = noises.shape[0], noises.shape[2]
        device, dtype = noises.device, noises.dtype

        # Step 1: Run generator on backward simulated noisy inputs
        with torch.no_grad():
            pred_x0, _, _ = self._run_generator(
                noises,
                prompt_embeds,
                cond_latents,
                plucker,
                #
                clean_latents,
                #
                C2W,
                fxfycxcy,
                depths,
                confs,
                images_f,
            )

        # Step 2: Compute the fake prediction
        min_t, max_t = int(self.opt.min_timestep_boundary * self.opt.num_train_timesteps), \
            int(self.opt.max_timestep_boundary * self.opt.num_train_timesteps)
        if not self.opt.is_teacher_causal:
            timesteps_id = torch.randint(min_t, max_t, (1,))  # (1,); batch share the same timestep for simpler time scheduler
            timesteps_id = timesteps_id.unsqueeze(1).repeat(B, f)  # (B, f)
        else:  # teacher / diffusion forcing
            assert f % self.opt.chunk_size == 0
            num_chunks = f // self.opt.chunk_size

            timesteps_id = torch.randint(min_t, max_t, (num_chunks,))  # (num_chunks,); each chunk in different noise level
            timesteps_id = timesteps_id.repeat_interleave(self.opt.chunk_size, dim=0).repeat(B, 1)  # (B, f); batch share the same timestep for simpler time scheduler
        timesteps = self.diffusion.scheduler.timesteps[timesteps_id].to(dtype=dtype, device=device)
        if cond_latents is not None and self.opt.teacher_first_latent_cond:
            timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

        critic_noises = torch.randn_like(pred_x0)
        noisy_latents = self.diffusion.scheduler.add_noise(
            pred_x0.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
            critic_noises.transpose(1, 2).flatten(0, 1),   # (B*f, D, h, w)
            timesteps.flatten(0, 1),                # (B*f,)
        ).detach().unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)
        targets = self.diffusion.scheduler.training_target(pred_x0, critic_noises)

        fake_model_outputs = self.fake_score(
            noisy_latents,
            timesteps,
            prompt_embeds,
            plucker=plucker,
        )

        # Step 3: Compute the denoising loss for the fake critic
        diffusion_loss = tF.mse_loss(fake_model_outputs.float(), targets.float(), reduction="none")  # (B, D, f, h, w)
        diffusion_loss = self.diffusion.scheduler.training_weight(timesteps.flatten(0, 1)).reshape(-1, 1, 1, 1) * \
            diffusion_loss.transpose(1, 2).flatten(0, 1)  # (B*f, D, h, w)
        diffusion_loss = diffusion_loss.unflatten(0, (B, f)).transpose(1, 2)  # (B, D, f, h, w)
        return diffusion_loss.mean()

    def generator_loss(self,
        noises: Tensor,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        clean_latents: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
        depths: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
        images_f: Optional[Tensor] = None,
    ):
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_x0, gradient_mask, da3_outputs = self._run_generator(
            noises,
            prompt_embeds,
            cond_latents,
            plucker,
            #
            clean_latents,
            #
            C2W,
            fxfycxcy,
            depths,
            confs,
            images_f,
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_grad_norm = \
            self._compute_distribution_matching_loss(
                pred_x0,
                prompt_embeds,
                negative_prompt_embeds,
                gradient_mask,
                cond_latents,
                plucker,
            )

        # (Optional) Step 3: DA3 outputs
        if da3_outputs is not None:
            ## CausVid
            if not self.use_self_forcing:
                if "timesteps" in da3_outputs:
                    if self.opt.da3_weight_type == "uniform":
                        da3_weights = 1.
                    elif self.opt.da3_weight_type == "diffusion":
                        da3_weights = self.diffusion.scheduler.training_weight(da3_outputs["timesteps"].flatten(0, 1))
                    elif self.opt.da3_weight_type == "inverse_timestep":
                        da3_weights = 1. / (da3_outputs["timesteps"].flatten(0, 1) + 0.1)
                else:
                    da3_weights = 1.

                if "depth_loss" not in da3_outputs:
                    assert depths is not None
                    depth_loss = self.depth_loss_fn(da3_outputs["depth"], depths, confs=da3_outputs["depth_conf"])  # (B, f)
                    da3_outputs["depth_loss"] = (da3_weights * depth_loss.flatten(0, 1)).mean()
                if "ray_loss" not in da3_outputs or "camera_loss" not in da3_outputs:
                    assert C2W is not None and fxfycxcy is not None
                    H, W = noises.shape[-2] * 8, noises.shape[-1] * 8  # `8`: hard-coded for Wan2.1
                    _, (ray_o, ray_d) = plucker_ray(H//2//self.opt.da3_down_ratio, W//2//self.opt.da3_down_ratio,
                        C2W.float(), fxfycxcy.float(), normalize_ray_d=False)
                    gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(noises.dtype)  # (B, f, 6, H/2, W/2)
                    gt_pose_enc = torch.cat([
                        C2W[:, :, :3, 3].float(),  # (B, f, 3)
                        mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
                        2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
                        2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
                    ], dim=-1).to(noises.dtype)  # (B, f, 9)
                        ## Compute geometry losses
                    ray_loss = self.ray_loss_fn(da3_outputs["ray"], gt_raymaps, confs=da3_outputs["ray_conf"])  # (B, f)
                    camera_loss = self.camera_loss_fn(da3_outputs["pose_enc"], gt_pose_enc)  # (B, f)
                    da3_outputs["ray_loss"] = (da3_weights * ray_loss.flatten(0, 1)).mean()
                    da3_outputs["camera_loss"] = (da3_weights * camera_loss.flatten(0, 1)).mean()

        return dmd_loss, dmd_grad_norm, pred_x0, da3_outputs

    def _compute_distribution_matching_loss(self,
        pred_x0: Tensor,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Tensor,
        gradient_mask: Optional[Tensor] = None,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
    ):
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        """
        B, f = pred_x0.shape[0], pred_x0.shape[2]
        device, dtype = pred_x0.device, pred_x0.dtype

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_t, max_t = int(self.opt.min_timestep_boundary * self.opt.num_train_timesteps), \
                int(self.opt.max_timestep_boundary * self.opt.num_train_timesteps)
            if not self.opt.is_teacher_causal:
                timesteps_id = torch.randint(min_t, max_t, (1,))  # (1,); batch share the same timestep for simpler time scheduler
                timesteps_id = timesteps_id.unsqueeze(1).repeat(B, f)  # (B, f)
            else:  # teacher / diffusion forcing
                assert f % self.opt.chunk_size == 0
                num_chunks = f // self.opt.chunk_size

                timesteps_id = torch.randint(min_t, max_t, (num_chunks,))  # (num_chunks,); each chunk in different noise level
                timesteps_id = timesteps_id.repeat_interleave(self.opt.chunk_size, dim=0).repeat(B, 1)  # (B, f); batch share the same timestep for simpler time scheduler
            timesteps = self.diffusion.scheduler.timesteps[timesteps_id].to(dtype=dtype, device=device)
            if cond_latents is not None and self.opt.teacher_first_latent_cond:
                timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

            noisy_latents = self.diffusion.scheduler.add_noise(
                pred_x0.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
                torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),  # (B*f, D, h, w)
                timesteps.flatten(0, 1)                 # (B*f,)
            ).detach().unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)

            # Step 2: Compute the KL grad
            grad = self._compute_kl_grad(
                noisy_latents,
                pred_x0,
                timesteps,
                prompt_embeds,
                negative_prompt_embeds,
                plucker,
            )

        # The gradient of `dmd_loss` w.r.t. `pred_x0` is `grad`
        if gradient_mask is not None:
            dmd_loss = 0.5 * tF.mse_loss(
                pred_x0.double()[gradient_mask],
                (pred_x0.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * tF.mse_loss(
                pred_x0.double(),
                (pred_x0.double() - grad.double()).detach(),
                reduction="mean",
            )
        return dmd_loss, grad.abs().mean().detach()

    def _compute_kl_grad(self,
        noisy_latents: Tensor,
        pred_x0: Tensor,
        timesteps: Tensor,
        prompt_embeds: Tensor,
        negative_prompt_embeds: Tensor,
        plucker: Optional[Tensor] = None,
        normalization: bool = True,
    ):
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        """
        # Step 1: Compute the fake score
        fake_model_outputs = self.fake_score(
            noisy_latents,
            timesteps,
            prompt_embeds,
            plucker=plucker,
        )

        if self.opt.fake_guidance_scale != 1.:
            fake_model_outputs_uncond = self.fake_score(
                noisy_latents,
                timesteps,
                negative_prompt_embeds,
                plucker=plucker,
            )
            fake_model_outputs = fake_model_outputs_uncond + self.opt.fake_guidance_scale * (
                fake_model_outputs - fake_model_outputs_uncond)
        fake_pred_x0 = self.diffusion._convert_flow_pred_to_x0(fake_model_outputs, noisy_latents, timesteps)

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        real_model_outputs = self.real_score(
            noisy_latents,
            timesteps,
            prompt_embeds,
            plucker=plucker,
        )
        if self.opt.real_guidance_scale != 1.:
            real_model_outputs_uncond = self.real_score(
                noisy_latents,
                timesteps,
                negative_prompt_embeds,
                plucker=plucker,
            )
            real_model_outputs = real_model_outputs_uncond + self.opt.real_guidance_scale * (
                real_model_outputs - real_model_outputs_uncond)
        real_pred_x0 = self.diffusion._convert_flow_pred_to_x0(real_model_outputs, noisy_latents, timesteps)

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (fake_pred_x0 - real_pred_x0)

        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (pred_x0 - real_pred_x0)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad

    def _run_generator(self,
        noises: Tensor,
        prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        clean_latents: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
        depths: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
        images_f: Optional[Tensor] = None,
    ):
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        """
        # TODO: handle generating long videos

        if self.use_self_forcing:
            pred_x0, da3_outputs = self._consistency_backward_simulation(
                noises,
                prompt_embeds,
                cond_latents,
                plucker,
                #
                C2W,
                fxfycxcy,
            )
        else:
            assert clean_latents is not None

            B, f = noises.shape[0], noises.shape[2]
            device, dtype = noises.device, noises.dtype

            # (Optional) Point cloud rendering
            if self.opt.input_pcrender:
                assert depths is not None and confs is not None and images_f is not None
                H, W = images_f.shape[3], images_f.shape[4]
                with torch.no_grad():
                    if not self.opt.is_causal:
                        all_render_images = []
                        for i in range(B):
                            points, colors = filter_da3_points(
                                images_f[i], depths[i], confs[i], C2W[i], fxfycxcy[i],
                                conf_thresh_percentile=self.opt.conf_thresh_percentile,
                                random_sample_ratio=self.opt.rand_pcrender_ratio,
                                min_num_points=self.opt.min_num_points,
                                max_num_points=self.opt.max_num_points,
                            )
                            render_images = render_pt3d_points(H, W, points, colors, C2W[i], fxfycxcy[i])  # (f, 3, H, W) in [0, 1]
                            all_render_images.append(render_images.to(dtype))
                        render_images = torch.stack(all_render_images, dim=0)  # (B, f, 3, H, W) in [0, 1]
                    else:  # teacher forcing style conditioning
                        all_render_images = []
                        for i in range(B):
                            assert f % self.opt.chunk_size == 0
                            num_chunks = f // self.opt.chunk_size
                            all_render_images_chunk, all_points, all_colors = [], None, None
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
                                render_images = render_pt3d_points(H, W, all_points, all_colors, C2W[i, next_chunk_idxs], fxfycxcy[i, next_chunk_idxs]).to(dtype)  # (f_chunk, 3, H, W) in [0, 1]
                                all_render_images_chunk.append(render_images)
                            all_render_images_chunk = torch.cat(all_render_images_chunk, dim=0)  # (f-f_chunk, 3, H, W)

                            if cond_latents is not None:
                                points, colors = filter_da3_points(
                                    images_f[i, 0:1], depths[i, 0:1], confs[i, 0:1], C2W[i, 0:1], fxfycxcy[i, 0:1],
                                    conf_thresh_percentile=self.opt.conf_thresh_percentile,
                                    random_sample_ratio=self.opt.rand_pcrender_ratio,
                                    min_num_points=self.opt.min_num_points,
                                    max_num_points=self.opt.max_num_points,
                                )
                                all_render_images_chunk = torch.cat([
                                    render_pt3d_points(H, W, points, colors, C2W[i, :self.opt.chunk_size], fxfycxcy[i, :self.opt.chunk_size]).to(dtype),
                                    all_render_images_chunk,
                                ], dim=0)  # (f, 3, H, W)
                            else:
                                all_render_images_chunk = torch.cat([
                                    torch.zeros_like(all_render_images_chunk[:self.opt.chunk_size, ...]),
                                    all_render_images_chunk,
                                ], dim=0)  # (f, 3, H, W)

                            all_render_images.append(all_render_images_chunk)
                        render_images = torch.stack(all_render_images, dim=0)  # (B, f, 3, H, W)
            else:
                render_images = None

            denoising_step_list = torch.tensor(self.opt.denoising_step_list, dtype=torch.long)
            if self.opt.warp_denoising_step:
                timesteps = torch.cat((self.diffusion.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                denoising_step_list = timesteps[self.opt.num_train_timesteps - denoising_step_list]

            if not self.opt.is_causal:
                timesteps_id = torch.randint(0, len(denoising_step_list), (1,))  # (1,); batch share the same timestep for simpler time scheduler
                timesteps_id = timesteps_id.unsqueeze(1).repeat(B, f)  # (B, f)
            else:  # teacher / diffusion forcing
                assert f % self.opt.chunk_size == 0
                num_chunks = f // self.opt.chunk_size

                timesteps_id = torch.randint(0, len(denoising_step_list), (num_chunks,))  # (num_chunks,); each chunk in different noise level
                timesteps_id = timesteps_id.repeat_interleave(self.opt.chunk_size, dim=0).repeat(B, 1)  # (B, f); batch share the same timestep for simpler time scheduler
            timesteps = denoising_step_list[timesteps_id].to(dtype=dtype, device=device)
            if cond_latents is not None and self.opt.teacher_first_latent_cond:
                timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

            noisy_latents = self.diffusion.scheduler.add_noise(
                clean_latents.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
                noises.transpose(1, 2).flatten(0, 1),        # (B*f, D, h, w)
                timesteps.flatten(0, 1),                 # (B*f,)
            ).unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)

            model_outputs = self.diffusion(
                noisy_latents,
                timesteps,
                prompt_embeds,
                plucker=plucker,
                C2W=C2W, fxfycxcy=fxfycxcy,  # for DA3
                extra_condition=render_images,
            )
            model_outputs, da3_outputs = \
                model_outputs if self.opt.load_da3 else (model_outputs, None)
            pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_latents, timesteps).to(dtype)
            if da3_outputs is not None:
                da3_outputs["timesteps"] = timesteps
            if render_images is not None:
                da3_outputs["images_render"] = render_images

        gradient_mask = None  # TODO: handle generating long videos
        # if self.opt.first_latent_cond:
        #     gradient_mask = torch.ones_like(pred_x0, dtype=torch.bool)
        #     gradient_mask[:, :, 0:1, :, :] = False  # do not compute gradient on the first latent frame

        return pred_x0, gradient_mask, da3_outputs

    def _consistency_backward_simulation(self,
        noises: Tensor,
        prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
    ):
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        if not self.opt.rolling:
            return self.inference_pipeline.inference_with_trajectory(
                noises,
                prompt_embeds,
                cond_latents,
                plucker,
                #
                C2W,
                fxfycxcy,
            )
        else:  # rolling forcing
            return self.inference_pipeline.inference_with_trajectory_rolling(
                noises,
                prompt_embeds,
                cond_latents,
                plucker,
                #
                C2W,
                fxfycxcy,
            )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP/DeepSpeed-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(self.opt, self.diffusion, self.current_vae_decoder)

    def _add_lora_to_fake_score(self, target_modules: List[str], lora_rank: int, lora_alpha: Optional[int] = None):
        if lora_alpha is None:
            lora_alpha = lora_rank

        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        self.fake_score.model = inject_adapter_in_model(lora_config, self.fake_score.model)
