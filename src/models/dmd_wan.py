from typing import *
from torch import Tensor

import os
import numpy as np
import torch
import torch.nn.functional as tF
from peft import LoraConfig, inject_adapter_in_model

from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options
from src.models.modules import WanDiffusionWrapper, WanVAEWrapper
from src.models.wan import Wan
from src.models.pipelines.self_forcing_training import SelfForcingTrainingPipeline
from src.utils.ema import EMAParams
from src.utils import plucker_ray, colorize_depth, filter_da3_points, render_pt3d_points, mv_interpolate


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DMD_Wan(Wan):
    def __init__(self, opt: Options, lazy: bool = False):
        super().__init__(opt, lazy=lazy)

        # Real score model for DMD
        self.real_score = WanDiffusionWrapper(
            opt.real_wan_dir,
            opt.num_train_timesteps,
            opt.num_inference_steps,
            opt.shift,
            0.,    # hard-coded `sigma_min`
            True,  # hard-coded `extra_one_step`
            #
            opt.teacher_input_plucker,
            opt.teacher_extra_condition_dim,
            #
            opt.use_gradient_checkpointing,
            opt.use_gradient_checkpointing_offload,
            #
            is_causal=opt.is_teacher_causal,
            sink_size=opt.sink_size,
            chunk_size=opt.chunk_size,
            max_attention_size=opt.max_attention_size,
            #
            skip_pretrained_weights=opt.teacher_path is not None,
        )
        if opt.teacher_path is not None and not lazy:
            state_dict = torch.load(opt.teacher_path, map_location="cpu", weights_only=True)
            if "generator_ema" in state_dict:
                self.real_score.load_state_dict(state_dict["generator_ema"])
            elif "generator" in state_dict:
                self.real_score.load_state_dict(state_dict["generator"])
            else:
                self.real_score.load_state_dict(state_dict)
        self.real_score.requires_grad_(False)
        self.real_score.eval()

        # Fake score model for DMD (skipped when DDT[-1] replaces it)
        if not opt.ddt_fake_score:
            self.fake_score = WanDiffusionWrapper(
                opt.fake_wan_dir,
                opt.num_train_timesteps,
                opt.num_inference_steps,
                opt.shift,
                0.,    # hard-coded `sigma_min`
                True,  # hard-coded `extra_one_step`
                #
                opt.teacher_input_plucker,
                opt.teacher_extra_condition_dim,
                #
                opt.use_gradient_checkpointing,
                opt.use_gradient_checkpointing_offload,
                #
                is_causal=opt.is_teacher_causal,
                sink_size=opt.sink_size,
                chunk_size=opt.chunk_size,
                max_attention_size=opt.max_attention_size,
                #
                skip_pretrained_weights=opt.fake_path is not None,
            )
            if opt.fake_path is not None and not lazy:
                state_dict = torch.load(opt.fake_path, map_location="cpu", weights_only=True)
                if "generator_ema" in state_dict:
                    self.fake_score.load_state_dict(state_dict["generator_ema"])
                elif "generator" in state_dict:
                    self.fake_score.load_state_dict(state_dict["generator"])
                else:
                    self.fake_score.load_state_dict(state_dict)
        else:
            self.fake_score = None

        # This will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Add LoRA in the diffusion model, will freeze all parameters except LoRA layers
        if opt.use_lora_in_fake_score and not opt.ddt_fake_score:
            self._add_lora_to_fake_score(
                target_modules=opt.lora_target_modules_in_fake_score.split(","),
                lora_rank=opt.lora_rank_in_fake_score,
            )
            # Load LoRA checkpoint if specified
            if opt.fake_lora_path is not None and not lazy:
                lora_state_dict = torch.load(opt.fake_lora_path, map_location="cpu", weights_only=True)
                self.load_fake_score_lora_weights(lora_state_dict, strict=True)

        # Set other trainable parameters except LoRA layers in the diffusion model
        if opt.more_trainable_fake_score_params is not None and not opt.ddt_fake_score:
            trainble_names = opt.more_trainable_fake_score_params.split(",")
            if opt.use_lora_in_fake_score:
                trainble_names.append("lora")
            for name, param in self.fake_score.named_parameters():
                _flag = False
                for trainble_name in trainble_names:
                    if trainble_name in name:
                        param.requires_grad_(True)
                        _flag = True
                        break
                if not _flag:
                    param.requires_grad_(False)

    def compute_loss(self,
        data: Dict[str, Any],
        dtype: torch.dtype = torch.float32,
        train_generator: bool = True,
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

        # CausVid or Self-Forcing
        self.use_self_forcing = use_self_forcing = np.random.rand() <= self.opt.self_forcing_prob
        use_diffusion_loss = np.random.rand() < self.opt.diffusion_loss_prob
        if not use_self_forcing or use_diffusion_loss:
            assert "image" in data

        if "image" in data:
            images = data["image"].to(device=device, dtype=dtype)  # (B, F, 3, H, W)
            (B, F, _, H, W) = images.shape
        else:
            B = len(data["prompt"])
            F, H, W = (self.opt.num_input_frames - 1) * actual_num_clips + 1, self.opt.input_res[0], self.opt.input_res[1]

        # Text encoder
        if self.text_encoder is not None:
            if self.prompt_list is None or not self.use_self_forcing or use_diffusion_loss or np.random.rand() >= self.opt.vidprom_prob:
                prompts = data["prompt"]  # a list of strings
            else:
                actual_num_clips = len(data["prompt"][0]) if isinstance(data["prompt"][0], list) else 1
                assert actual_num_clips == 1  # VidProm only supports single clip
                prompts = np.random.choice(self.prompt_list, B, replace=False).tolist()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                self.text_encoder.eval()
                prompt_embeds = self._encode_prompt_batch(prompts)  # (B, N=512, D') or (B, num_clips, N=512, D')
                negative_prompt_embeds = self._build_negative_prompt_embeds(B, actual_num_clips)  # (B, N=512, D') or (B, num_clips, N=512, D')
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
        cond_latents = cond_latents if torch.rand(1).item() < self.opt.random_i2v_prob else None

        # (Optional) Camera & depth
        idxs = torch.arange(0, F, 4).to(device=device, dtype=torch.long)
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
        if "image" in data:
            images_f = data["image"].to(device=device, dtype=dtype)[:, idxs, ...]  # (B, f, 3, H, W)
        else:
            images_f = None

        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)

        # Shared noises for generator and critic to enable `pred_x0` reuse
        shared_noises = torch.randn_like(latents)

        # DMD + (Optional) Diffusion
        if train_generator:
            ## 1. Generator loss
            generator_loss, dmd_grad_norm, pred_x0, da3_outputs = \
                self.generator_loss(
                    shared_noises,
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
                    #
                    clip_latent_lens=clip_latent_lens,
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

            ## 2. (Optional) Diffusion loss
            if use_diffusion_loss:
                diffusion_loss, pred_x0_diffusion, da3_outputs_diffusion = \
                    self.diffusion_loss(
                        latents,
                        prompt_embeds,
                        cond_latents,
                        plucker,
                        #
                        C2W=C2W,
                        fxfycxcy=fxfycxcy,
                        depths=depths,
                        confs=confs,
                        images_f=images_f,
                        #
                        clip_latent_lens=clip_latent_lens,
                    )
                outputs["diffusion_loss"] = diffusion_loss

                if da3_outputs_diffusion is not None:
                    if "depth_loss" in da3_outputs_diffusion:
                        outputs["depth_loss_diffusion"] = da3_outputs_diffusion["depth_loss"]
                        diffusion_loss = diffusion_loss + da3_outputs_diffusion["depth_loss"]
                    if "ray_loss" in da3_outputs_diffusion:
                        outputs["ray_loss_diffusion"] = da3_outputs_diffusion["ray_loss"]
                        diffusion_loss = diffusion_loss + da3_outputs_diffusion["ray_loss"]
                    if "camera_loss" in da3_outputs_diffusion:
                        outputs["camera_loss_diffusion"] = da3_outputs_diffusion["camera_loss"]
                        diffusion_loss = diffusion_loss + da3_outputs_diffusion["camera_loss"]
            else:
                pred_x0_diffusion = None
                da3_outputs_diffusion = None
                diffusion_loss = 0.

            # Collect generator-side losses for separate backward pass
            generator_total_loss = generator_loss + self.opt.diffusion_loss_weight * diffusion_loss
        else:
            pred_x0, pred_x0_diffusion = None, None
            da3_outputs, da3_outputs_diffusion = None, None
            generator_total_loss = None

        ## 3. Critic loss — reuse generator's `pred_x0` when available to skip redundant self-forcing
        outputs["critic_loss"] = critic_loss = \
            self.critic_loss(
                shared_noises,
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
                #
                clip_latent_lens=clip_latent_lens,
                #
                cached_pred_x0=pred_x0 if train_generator else None,
            )

        # Return separate losses for sequential backward passes to reduce peak memory
        # Each loss is backward-ed independently so their activation graphs don't overlap
        losses = []
        if generator_total_loss is not None:
            losses.append(generator_total_loss)
        losses.append(self.opt.critic_loss_weight * critic_loss)
        outputs["losses"] = losses

        # Scalar `loss` for logging only (detached sum)
        outputs["loss"] = sum(l.detach() for l in losses)

        # # For visualizaiton
        # if is_eval:
        #     if pred_x0 is not None:
        #         outputs["images_predx0"] = (self.decode_latent(pred_x0, vae).clamp(-1., 1.) + 1.) / 2.
        #     if pred_x0_diffusion is not None:
        #         outputs["images_predx0_diffusion"] = (self.decode_latent(pred_x0_diffusion, vae).clamp(-1., 1.) + 1.) / 2.
        #     if "image" in data:
        #         outputs["images_input"] = data["image"].to(device)
        #     ## DMD
        #     if da3_outputs is not None:
        #         if "depth" in da3_outputs:
        #             outputs["images_pred_depth"] = colorize_depth(1./da3_outputs["depth"], batch_mode=True)
        #         if "images_render" in da3_outputs:
        #             outputs["images_render"] = da3_outputs["images_render"]
        #         if "images_render_depth" in da3_outputs:
        #             outputs["images_render_depth"] = da3_outputs["images_render_depth"]
        #     ## (Optional) Diffusion
        #     if da3_outputs_diffusion is not None:
        #         if "depth" in da3_outputs_diffusion:
        #             outputs["images_pred_depth_diffusion"] = colorize_depth(1./da3_outputs_diffusion["depth"], batch_mode=True)
        #         if "images_render" in da3_outputs_diffusion:
        #             outputs["images_render_diffusion"] = da3_outputs_diffusion["images_render"]
        #         if "images_render_depth" in da3_outputs_diffusion:
        #             outputs["images_render_depth_diffusion"] = da3_outputs_diffusion["images_render_depth"]

        return outputs

    ################################ Helper functions ################################

    def diffusion_loss(self,
        clean_latents: Tensor,
        prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
        depths: Optional[Tensor] = None,
        confs: Optional[Tensor] = None,
        images_f: Optional[Tensor] = None,
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
    ):
        noises = torch.randn_like(clean_latents)
        B, f = noises.shape[0], noises.shape[2]
        device, dtype = noises.device, noises.dtype

        # (Optional) Point cloud rendering
        if self.opt.input_pcrender:
            assert depths is not None and confs is not None and images_f is not None
            H, W = images_f.shape[3], images_f.shape[4]
            if self.opt.da3_down_ratio != 1:
                images_f = mv_interpolate(images_f,
                    size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)
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
        if cond_latents is not None and self.opt.first_latent_cond:
            timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

        if self.opt.no_noise_for_da3:
            timesteps = torch.zeros_like(timesteps)  # only use clean latents for DA3 supervision
        else:  # noisy inputs
            if self.opt.is_causal:
                chunk_id = np.random.randint(1, num_chunks)  # randomly choice one chunk to be noisy, others are clean
                timesteps = torch.cat([
                    torch.zeros_like(timesteps[:, :chunk_id*self.opt.chunk_size]),
                    timesteps[:, chunk_id*self.opt.chunk_size:(chunk_id+1)*self.opt.chunk_size],
                    torch.zeros_like(timesteps[:, (chunk_id+1)*self.opt.chunk_size:]),
                ], dim=1)  # (B, f)

        noisy_latents = self.diffusion.scheduler.add_noise(
            clean_latents.transpose(1, 2).flatten(0, 1),  # (B*f, D, h, w)
            noises.transpose(1, 2).flatten(0, 1),        # (B*f, D, h, w)
            timesteps.flatten(0, 1),                 # (B*f,)
        ).unflatten(0, (B, f)).transpose(1, 2).to(dtype)  # (B, D, f, h, w)
        targets = self.diffusion.scheduler.training_target(clean_latents, noises)

        model_outputs = self.diffusion(
            noisy_latents,
            timesteps,
            prompt_embeds,
            plucker=plucker,
            C2W=C2W, fxfycxcy=fxfycxcy,  # for DA3
            extra_condition=input_extra_condition,
            #
            clean_x=clean_latents if self.opt.use_teacher_forcing else None,
            #
            clip_latent_lens=clip_latent_lens,  # for multi-clip generation
            #
            ddt_index=1 if self.opt.ddt_diffusion_loss else 0,  # first or second DDT head for diffusion loss
        )
        model_outputs, da3_outputs = \
            model_outputs if self.opt.load_da3 else (model_outputs, None)
        pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_latents, timesteps).to(dtype)

        diffusion_loss = tF.mse_loss(model_outputs.float(), targets.float(), reduction="none")  # (B, D, f, h, w)
        diffusion_loss = (self.diffusion.scheduler.training_weight(timesteps.flatten(0, 1)).reshape(-1, 1, 1, 1) + 1e-3) * \
            diffusion_loss.transpose(1, 2).flatten(0, 1)  # (B*f, D, h, w)
        diffusion_loss = diffusion_loss.unflatten(0, (B, f)).transpose(1, 2)  # (B, D, f, h, w)

        if self.opt.no_noise_for_da3:
            da3_weights = 1.
        else:
            if self.opt.da3_weight_type == "uniform":
                da3_weights = 1.
            elif self.opt.da3_weight_type == "diffusion":
                da3_weights = self.diffusion.scheduler.training_weight(timesteps.flatten(0, 1))
            elif self.opt.da3_weight_type == "inverse_timestep":
                da3_weights = 1. / (timesteps.flatten(0, 1) + 0.1)
            else:
                da3_weights = 1.

        if da3_outputs is not None:
            if render_images is not None:
                da3_outputs["images_render"] = render_images

            assert depths is not None
            depth_loss = self.depth_loss_fn(da3_outputs["depth"], depths, confs=da3_outputs["depth_conf"])  # (B, f)
            da3_outputs["depth_loss"] = (da3_weights * depth_loss.flatten(0, 1)).mean()

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

        return diffusion_loss.mean(), pred_x0, da3_outputs

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
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
        #
        cached_pred_x0: Optional[Tensor] = None,  # reuse `pred_x0` from generator to skip self-forcing replay
    ):
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        """
        B, f = noises.shape[0], noises.shape[2]
        device, dtype = noises.device, noises.dtype

        # Step 1: Obtain fake videos — reuse from generator if available, otherwise run generator
        if cached_pred_x0 is not None:
            pred_x0 = cached_pred_x0.detach()
        else:
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
                    #
                    clip_latent_lens,
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

        if self.opt.ddt_fake_score:
            fake_model_outputs = self.diffusion(
                noisy_latents,
                timesteps,
                prompt_embeds,
                plucker=plucker,
                #
                clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                #
                ddt_index=-1,  # last DDT head as fake score
            )
        else:
            fake_model_outputs = self.fake_score(
                noisy_latents,
                timesteps,
                prompt_embeds,
                plucker=plucker,
                #
                clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
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
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
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
            #
            clip_latent_lens,
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
                #
                clip_latent_lens,
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
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
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
                #
                clip_latent_lens,
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
        #
        clip_latent_lens: Optional[Tensor] = None,
        #
        normalization: bool = True,
    ):
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        """
        # Step 1: Compute the fake score (DDT[-1] or independent fake_score)
        if self.opt.ddt_fake_score:
            fake_model_outputs = self.diffusion(
                noisy_latents,
                timesteps,
                prompt_embeds,
                plucker=plucker,
                #
                clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                #
                ddt_index=-1,  # last DDT head as fake score
            )
        else:
            fake_model_outputs = self.fake_score(
                noisy_latents,
                timesteps,
                prompt_embeds,
                plucker=plucker,
                #
                clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
            )

        if self.opt.fake_guidance_scale != 1.:
            if self.opt.ddt_fake_score:
                fake_model_outputs_uncond = self.diffusion(
                    noisy_latents,
                    timesteps,
                    negative_prompt_embeds,
                    plucker=plucker,
                    #
                    clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    #
                    ddt_index=-1,  # last DDT head as fake score
                )
            else:
                fake_model_outputs_uncond = self.fake_score(
                    noisy_latents,
                    timesteps,
                    negative_prompt_embeds,
                    plucker=plucker,
                    #
                    clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                    #
                    clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                )
            fake_model_outputs = fake_model_outputs_uncond + self.opt.fake_guidance_scale * (
                fake_model_outputs - fake_model_outputs_uncond)
            del fake_model_outputs_uncond  # free intermediate tensors
        fake_pred_x0 = self.diffusion._convert_flow_pred_to_x0(fake_model_outputs, noisy_latents, timesteps)
        del fake_model_outputs  # free intermediate tensors

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        real_model_outputs = self.real_score(
            noisy_latents,
            timesteps,
            prompt_embeds,
            plucker=plucker,
            #
            clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
            #
            clip_latent_lens=clip_latent_lens,  # for multi-clip generation
        )
        if self.opt.real_guidance_scale != 1.:
            real_model_outputs_uncond = self.real_score(
                noisy_latents,
                timesteps,
                negative_prompt_embeds,
                plucker=plucker,
                #
                clean_x=pred_x0 if self.opt.teacher_use_teacher_forcing else None,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
            )
            real_model_outputs = real_model_outputs_uncond + self.opt.real_guidance_scale * (
                real_model_outputs - real_model_outputs_uncond)
            del real_model_outputs_uncond  # free intermediate tensors
        real_pred_x0 = self.diffusion._convert_flow_pred_to_x0(real_model_outputs, noisy_latents, timesteps)
        del real_model_outputs  # free intermediate tensors

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (fake_pred_x0 - real_pred_x0)
        del fake_pred_x0  # free intermediate tensors

        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (pred_x0 - real_pred_x0)
            del real_pred_x0  # free intermediate tensors
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            del p_real  # free intermediate tensors
            grad = grad / normalizer
        else:
            del real_pred_x0  # free intermediate tensors
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
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
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
                #
                clip_latent_lens,
            )
        else:
            assert clean_latents is not None

            B, f = noises.shape[0], noises.shape[2]
            device, dtype = noises.device, noises.dtype

            # (Optional) Point cloud rendering
            if self.opt.input_pcrender:
                assert depths is not None and confs is not None and images_f is not None
                H, W = images_f.shape[3], images_f.shape[4]
                if self.opt.da3_down_ratio != 1:
                    images_f = mv_interpolate(images_f,
                        size=(H//self.opt.da3_down_ratio, W//self.opt.da3_down_ratio), mode="bilinear", align_corners=False)
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
                extra_condition=input_extra_condition,
                #
                clean_x=clean_latents if self.opt.use_teacher_forcing else None,
                #
                clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                ddt_index=0,  # the first DDT head
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
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
    ):
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noises,
            prompt_embeds,
            cond_latents,
            plucker,
            #
            C2W,
            fxfycxcy,
            #
            clip_latent_lens,
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP/DeepSpeed-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(self.opt, self.diffusion, self.current_vae_decoder)

    def _add_lora_to_fake_score(self, target_modules: List[str], lora_rank: int, lora_alpha: Optional[int] = None):
        assert not self.opt.ddt_fake_score  # LoRA is not supported for DDT fake score
        if lora_alpha is None:
            lora_alpha = lora_rank

        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        self.fake_score = inject_adapter_in_model(lora_config, self.fake_score)

        # Freeze all base model parameters, only train LoRA weights
        for name, param in self.fake_score.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def get_fake_score_lora_state_dict(self):
        """Get only LoRA parameters from fake_score for saving.

        This method is FSDP-aware and will gather sharded parameters from all ranks.
        When using FSDP, this should be called on all ranks, but only rank 0 will
        receive the full state dict.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Check if the model is wrapped with FSDP
        if isinstance(self.fake_score, FSDP):
            # Use FSDP API to gather full state dict on rank 0
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fake_score, StateDictType.FULL_STATE_DICT, cfg):
                full_state_dict = self.fake_score.state_dict()

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
            for name, param in self.fake_score.named_parameters():
                if "lora_" in name:
                    lora_state_dict[name] = param.cpu().clone()
            return lora_state_dict

    def load_fake_score_lora_weights(self, lora_state_dict: dict, strict: bool = True):
        """Load LoRA weights into the fake_score model."""
        missing_keys, unexpected_keys = [], []
        for name, param in lora_state_dict.items():
            if name in dict(self.fake_score.named_parameters()):
                dict(self.fake_score.named_parameters())[name].data.copy_(param)
            else:
                unexpected_keys.append(name)

        if strict:
            for name, param in self.fake_score.named_parameters():
                if "lora_" in name and name not in lora_state_dict:
                    missing_keys.append(name)

            if missing_keys or unexpected_keys:
                error_msg = f"Error loading fake_score LoRA weights:\n"
                if missing_keys:
                    error_msg += f"Missing keys: {missing_keys}\n"
                if unexpected_keys:
                    error_msg += f"Unexpected keys: {unexpected_keys}\n"
                raise RuntimeError(error_msg)

    def merge_fake_score_lora_weights(self):
        """Merge LoRA weights into the fake_score base model and remove LoRA adapters.

        This is useful when you want to:
        1. Create a standalone fake_score model with LoRA weights baked in
        2. Train a new LoRA on top of the merged weights

        After calling this method, the fake_score model will no longer have LoRA adapters,
        and all LoRA weights will be merged into the base model parameters.

        Returns:
            None
        """
        if not hasattr(self.fake_score, 'merge_and_unload'):
            raise RuntimeError("fake_score model does not have LoRA adapters. Cannot merge.")

        # `merge_and_unload()` returns a new model with LoRA weights merged
        self.fake_score = self.fake_score.merge_and_unload()
