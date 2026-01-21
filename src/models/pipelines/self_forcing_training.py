from typing import *
from torch import Tensor
from src.models.wan import WanDiffusionWrapper, WanDiffusionDA3Wrapper
from src.models.networks import VAEDecoderWrapper, TAEHV
from src.models.networks.da3_wrapper import DA3Wrapper

import torch
import torch.distributed as dist
import torch.nn.functional as tF
from einops import rearrange

from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options
from src.models.losses import XYZLoss, DepthLoss, CameraLoss
from src.utils.constant import ZERO_VAE_CACHE
from src.utils import plucker_ray, filter_da3_points, render_pt3d_points


class SelfForcingTrainingPipeline:
    def __init__(self,
        opt: Options,
        diffusion: WanDiffusionWrapper | WanDiffusionDA3Wrapper,
        current_vae_decoder: Optional[VAEDecoderWrapper | TAEHV] = None,
        da3_wrapper: Optional[DA3Wrapper] = None,
    ):
        super().__init__()

        self.opt = opt
        self.diffusion = diffusion
        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)

        self.denoising_step_list = torch.tensor(opt.denoising_step_list, dtype=torch.long)
        if opt.warp_denoising_step:
            timesteps = torch.cat((diffusion.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[self.opt.num_train_timesteps - self.denoising_step_list]
        self.denoising_step_list = torch.cat([self.denoising_step_list, torch.tensor([0], dtype=torch.float32)])  # add the last step

        self.kv_cache_pos = None
        self.crossattn_cache_pos = None
        self.kv_cache_pos_da3 = None

        self.ray_loss_fn, self.depth_loss_fn, self.camera_loss_fn = \
            XYZLoss(opt), DepthLoss(opt), CameraLoss(opt)

        self.current_vae_decoder = current_vae_decoder
        self.da3_wrapper = da3_wrapper

    def generate_and_sync_list(self, num_chunks: int, num_denoising_steps: int, device: torch.device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_chunks,),
                device=device
            )
            if self.opt.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_chunks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(self,
        noises: Tensor,
        prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
    ):
        B, _, f, h, w = noises.shape
        device, dtype = noises.device, noises.dtype

        if cond_latents is not None:
            noises = torch.cat([cond_latents, noises[:, :, 1:, ...]], dim=2)
        outputs = torch.zeros_like(noises)

        # Set KV cache
        if self.opt.is_causal:
            self._initialize_kv_cache(B, dtype, device)
            self._initialize_crossattn_cache(B, dtype, device)
            if self.opt.memory_num_tokens > 0:
                memory_tokens = self.diffusion.init_state(torch.arange(self.opt.memory_num_tokens, device=device)).expand(B, -1, -1)
            else:
                memory_tokens = None
        else:
            memory_tokens = None

        # Auto-regression steps
        assert f % self.opt.chunk_size == 0
        num_chunks = f // self.opt.chunk_size
        frame_seqlen = h * w // 4  # `4`: hard-coded for 2x2 patch embedding in DiT
        exit_flags = self.generate_and_sync_list(num_chunks, len(self.denoising_step_list)-1, device)

        # (Optional) Point cloud rendering
        if self.opt.input_pcrender:
            assert self.opt.load_da3

            if cond_latents is not None:
                raise NotImplementedError  # TODO
            else:
                render_images = torch.zeros((B, self.opt.chunk_size, 3, h*8, w*8), dtype=dtype, device=device)  # `8`: hard-coded for Wan2.1
                render_masks = torch.zeros((B, self.opt.chunk_size, h*8, w*8), dtype=dtype, device=device)
                render_loss = 0.
            render_images_vis = render_images

            if self.opt.load_tae:
                vae_cache = None
            else:
                vae_cache = ZERO_VAE_CACHE
                for i in range(len(vae_cache)):
                    vae_cache[i] = vae_cache[i].to(device=device, dtype=dtype)
        else:
            render_images = None

        # Temporal denoising loop
        all_da3_outputs, all_points, all_colors, images_f, all_timesteps = [None] * num_chunks, [None] * B, [None] * B, [], []
        for chunk_idx in range(num_chunks):
            this_chunk_latents = noises[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            if self.opt.input_plucker:
                this_chunk_plucker = plucker[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            else:
                this_chunk_plucker = None
            this_chunk_C2W = C2W[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            this_chunk_fxfycxcy = fxfycxcy[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]

            # Spatial denoising loop
            for ti, timestep in enumerate(self.denoising_step_list[:-1]):
                # Only backprop at the randomly selected timestep (consistent across all ranks)
                if self.opt.same_step_across_chunks:
                    exit_flag = (ti == exit_flags[0])
                else:
                    exit_flag = (ti == exit_flags[chunk_idx])

                timesteps = timestep[None, None].repeat(B, self.opt.chunk_size).to(dtype=dtype, device=device)
                if chunk_idx == 0 and cond_latents is not None:
                    timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

                # Diffusion model
                if not exit_flag:
                    with torch.no_grad():
                        model_outputs = self.diffusion(
                            this_chunk_latents,
                            timesteps,
                            prompt_embeds,
                            plucker=this_chunk_plucker,
                            C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                            extra_condition=render_images,
                            #
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                            #
                            memory_tokens=memory_tokens,
                            #
                            kv_cache_da3=self.kv_cache_pos_da3,
                            current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                        )
                        if memory_tokens is not None:
                            model_outputs, memory_tokens_tmp = model_outputs  # NOTE: NOT update `memory_tokens` here

                        model_outputs, da3_outputs = \
                            model_outputs if self.opt.load_da3 else (model_outputs, None)

                        next_timesteps = self.denoising_step_list[ti + 1] * torch.ones_like(timesteps)
                        if chunk_idx == 0 and cond_latents is not None:
                            next_timesteps = torch.cat([torch.zeros_like(next_timesteps[:, :1]), next_timesteps[:, 1:]], dim=1)

                        if self.opt.deterministic_inference:
                            this_chunk_latents = (
                                this_chunk_latents.transpose(1, 2).flatten(0, 1) +
                                (
                                    model_outputs.transpose(1, 2).flatten(0, 1) *
                                    (next_timesteps - timesteps).flatten(0, 1).reshape(-1, 1, 1, 1) / self.opt.num_train_timesteps
                                )
                            ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)  # (B, D, f_chunk, h, w)
                        else:
                            pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps).to(dtype)
                            this_chunk_latents = self.diffusion.scheduler.add_noise(
                                pred_x0.transpose(1, 2).flatten(0, 1),
                                torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                                next_timesteps.flatten(0, 1),
                            ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)  # (B, D, f_chunk, h, w)
                else:
                    # For getting real output
                    model_outputs = self.diffusion(
                        this_chunk_latents,
                        timesteps,
                        prompt_embeds,
                        plucker=this_chunk_plucker,
                        C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                        extra_condition=render_images,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        memory_tokens=memory_tokens,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                    )
                    if memory_tokens is not None:
                        model_outputs, memory_tokens_tmp = model_outputs  # NOTE: NOT update `memory_tokens` here

                    model_outputs, da3_outputs = \
                        model_outputs if self.opt.load_da3 else (model_outputs, None)

                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps).to(dtype)
                    break

            # Record this chunk generated latents
            outputs[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...] = pred_x0
            all_da3_outputs[chunk_idx] = da3_outputs
            all_timesteps.append(timesteps)

            # Rerun with timestep `context_timestep` to update KV cache
            if self.opt.is_causal and chunk_idx < num_chunks - 1:
                context_timesteps = self.opt.context_noise * torch.ones_like(timesteps)
                if chunk_idx == 0 and cond_latents is not None:
                    context_timesteps = torch.cat([torch.zeros_like(context_timesteps[:, :1]), context_timesteps[:, 1:]], dim=1)

                pred_x0 = self.diffusion.scheduler.add_noise(  # add context noise
                    pred_x0.transpose(1, 2).flatten(0, 1),
                    torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                    context_timesteps.flatten(0, 1),
                ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)
                with torch.no_grad():
                    model_outputs = self.diffusion(
                        pred_x0,
                        context_timesteps,
                        prompt_embeds,
                        plucker=this_chunk_plucker,
                        C2W=this_chunk_C2W, fxfycxcy=this_chunk_fxfycxcy,  # for DA3
                        extra_condition=render_images,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        memory_tokens=memory_tokens,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                    )
                    if memory_tokens is not None:
                        model_outputs, memory_tokens = model_outputs  # NOTE: update `memory_tokens` here

            # (Optional) Decode images
            if self.opt.input_pcrender:
                assert self.current_vae_decoder is not None

                with torch.enable_grad() if self.opt.render_loss_in_sf else torch.no_grad():
                    if self.opt.load_tae:
                        if vae_cache is None:
                            vae_cache = pred_x0
                        else:
                            pred_x0 = torch.cat([vae_cache, pred_x0], dim=2)
                            vae_cache = pred_x0[:, :, -3:, :, :]
                        current_images_f = self.current_vae_decoder.decode(pred_x0)
                        if chunk_idx == 0:
                            current_images_f = current_images_f[:, 3:, :, :, :]  # skip the first 3 frames of first block
                        else:
                            current_images_f = current_images_f[:, 12:, :, :, :]
                    else:
                        current_images_f, vae_cache = self.current_vae_decoder(pred_x0, *vae_cache)
                        current_images_f = (current_images_f.clamp(-1., 1.) + 1.) / 2.
                        if chunk_idx == 0:
                            current_images_f = current_images_f[:, 3:, :, :, :]  # skip the first 3 frames of first block
                    if chunk_idx == 0:
                        _idxs = torch.arange(0, current_images_f.shape[1], 4).to(device=device, dtype=torch.long)
                    else:
                        _idxs = torch.arange(3, current_images_f.shape[1], 4).to(device=device, dtype=torch.long)
                    current_images_f = current_images_f[:, _idxs, :, :]  # (B, f_chunk, 3, H, W)
                    assert current_images_f.shape[1] == self.opt.chunk_size
                    images_f.append(current_images_f)

                if self.opt.render_loss_in_sf:
                    render_loss = render_loss + (
                        tF.mse_loss(current_images_f, render_images, reduction="none") * render_masks.unsqueeze(2)
                    ).sum() / (render_masks.sum() * 3 + 1e-8)

                # (Optional) Update render images for next chunks
                if chunk_idx < num_chunks - 1:
                    assert self.opt.load_da3

                    current_depths = all_da3_outputs[chunk_idx]["depth"]  # (B, f_chunk, H, W)
                    current_confs = all_da3_outputs[chunk_idx]["depth_conf"]  # (B, f_chunk, H, W)
                    current_C2W = all_da3_outputs[chunk_idx]["C2W"]  # (B, f_chunk, 4, 4)
                    current_fxfycxcy = all_da3_outputs[chunk_idx]["fxfycxcy"]  # (B, f_chunk, 4)

                    all_render_images = []
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
                            with torch.no_grad():
                                render_images = render_pt3d_points(h*8, w*8, all_points[i], all_colors[i],  # `*8`: hard-coded for Wan2.1
                                    C2W[i, (chunk_idx + 1) * self.opt.chunk_size:(chunk_idx + 2) * self.opt.chunk_size, ...],
                                    fxfycxcy[i, (chunk_idx + 1) * self.opt.chunk_size:(chunk_idx + 2) * self.opt.chunk_size, ...],
                                ).to(dtype)
                            render_masks = (render_images > 10/255).to(dtype)  # consider pixels with value > 10 as valid
                        else:  # no valid points
                            render_images = torch.zeros((self.opt.chunk_size, 3, h*8, w*8), dtype=dtype, device=device)
                            render_masks = torch.zeros((self.opt.chunk_size, h*8, w*8), dtype=dtype, device=device)
                        all_render_images.append(render_images)
                    render_images = torch.stack(all_render_images, dim=0)  # (B, f_chunk, 3, H, W)
                    render_images_vis = torch.cat([render_images_vis, render_images], dim=1)

        if self.opt.load_da3:
            assert da3_outputs is not None
            da3_outputs = {
                k: torch.cat([all_da3_outputs[i][k] for i in range(num_chunks)], dim=1)
                for k in all_da3_outputs[0].keys()
            }

        if self.opt.da3_loss_in_sf:
            all_timesteps = torch.cat(all_timesteps, dim=1)  # (B, f)
            if self.opt.da3_weight_type == "uniform":
                da3_weights = 1.
            elif self.opt.da3_weight_type == "diffusion":
                da3_weights = self.diffusion.scheduler.training_weight(all_timesteps.flatten(0, 1))
            elif self.opt.da3_weight_type == "inverse_timestep":
                da3_weights = 1. / (all_timesteps.flatten(0, 1) + 0.1)
            else:
                da3_weights = 1.

            # assert self.da3_wrapper is not None
            # images_f = torch.cat(images_f, dim=1).to(dtype)  # (B, f, 3, H, W)
            # assert images_f.shape[1] == f

            # # Run DA3 wrapper for depth loss
            # with torch.no_grad():
            #     images_f = rearrange(tF.interpolate(
            #         rearrange(images_f, "b f ... -> (b f) ..."),
            #         size=(280, 504), mode="bilinear", align_corners=False  # TODO: hard-coded res
            #     ), "(b f) c h w -> b f c h w", b=B)
            #     da3_wrapper_outputs = self.da3_wrapper(images_f)
            #     da3_wrapper_outputs["depth"] = rearrange(tF.interpolate(
            #         rearrange(da3_wrapper_outputs["depth"], "b f ... -> (b f) ...").unsqueeze(1),
            #         size=(288, 512), mode="bilinear", align_corners=False  # TODO: hard-coded res
            #     ), "(b f) ... -> b f ...", b=B).squeeze(2)

            # da3_outputs["depth_da3_wrapper"] = da3_wrapper_outputs["depth"]
            # da3_outputs["depth_loss"] = (da3_weights * self.depth_loss_fn(
            #     da3_outputs["depth"], da3_wrapper_outputs["depth"], confs=da3_outputs["depth_conf"]).flatten(0, 1)).mean()

            assert C2W is not None and fxfycxcy is not None
            H, W = self.opt.input_res

            # Get ground-truth geometry labels
            _, (ray_o, ray_d) = plucker_ray(H//2//self.opt.da3_down_ratio, W//2//self.opt.da3_down_ratio,
                C2W.float(), fxfycxcy.float(), normalize_ray_d=False)
            gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
            gt_pose_enc = torch.cat([
                C2W[:, :, :3, 3].float(),  # (B, f, 3)
                mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
                2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
                2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
            ], dim=-1).to(dtype)  # (B, f, 9)

            # Compute geometry losses
            ray_loss = (da3_weights * self.ray_loss_fn(da3_outputs["ray"], gt_raymaps, confs=da3_outputs["ray_conf"])).flatten(0, 1)  # (B*f,)
            camera_loss = (da3_weights * self.camera_loss_fn(da3_outputs["pose_enc"], gt_pose_enc)).flatten(0, 1)
            # camera_loss += (da3_weights * self.camera_loss_fn(da3_wrapper_outputs["pose_enc"], gt_pose_enc)).flatten(0, 1)  # (B*f,)
            da3_outputs["ray_loss"] = ray_loss.mean()
            da3_outputs["camera_loss"] = camera_loss.mean()
            if self.opt.input_pcrender and self.opt.render_loss_in_sf:
                da3_outputs["render_loss"] = render_loss / num_chunks

        if render_images is not None:
            da3_outputs["images_render"] = render_images_vis

        return outputs, da3_outputs

    def inference_with_trajectory_rolling(self,
        noises: Tensor,
        prompt_embeds: Tensor,
        cond_latents: Optional[Tensor] = None,
        plucker: Optional[Tensor] = None,
        #
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
    ):
        B, _, f, h, w = noises.shape
        device, dtype = noises.device, noises.dtype

        outputs = torch.zeros_like(noises)
        if self.opt.load_da3:
            all_da3_outputs = {
                "depth": torch.zeros((B, f, h*8//self.opt.da3_down_ratio, w*8//self.opt.da3_down_ratio), dtype=dtype, device=device),
                "depth_conf": torch.zeros((B, f, h*8//self.opt.da3_down_ratio, w*8//self.opt.da3_down_ratio), dtype=dtype, device=device),
                "ray": torch.zeros((B, f, 6, h*4//self.opt.da3_down_ratio, w*4//self.opt.da3_down_ratio), dtype=dtype, device=device),
                "ray_conf": torch.zeros((B, f, h*4//self.opt.da3_down_ratio, w*4//self.opt.da3_down_ratio), dtype=dtype, device=device),
                "pose_enc": torch.zeros((B, f, 9), dtype=dtype, device=device),
                #
                "C2W": torch.zeros((B, f, 4, 4), dtype=dtype, device=device),
                "fxfycxcy": torch.zeros((B, f, 4), dtype=dtype, device=device),
            }

        # Set KV cache
        self._initialize_kv_cache(B, dtype, device)
        self._initialize_crossattn_cache(B, dtype, device)

        # Auto-regression steps
        assert f % self.opt.chunk_size == 0
        num_chunks = f // self.opt.chunk_size
        frame_seqlen = h * w // 4  # `4`: hard-coded for 2x2 patch embedding in DiT

        # Rolling Forcing
        rolling_window_length = num_denoising_steps = len(self.denoising_step_list[:-1])
        window_start_chunks, window_end_chunks = [], []
        window_num = num_chunks + rolling_window_length - 1
        for win_idx in range(window_num):
            start_chunk = max(0, win_idx - rolling_window_length + 1)
            end_chunk = min(num_chunks - 1, win_idx)
            window_start_chunks.append(start_chunk)
            window_end_chunks.append(end_chunk)

        # `exit_flag` indicates the window at which the model will backpropagate gradients
        exit_flag = torch.randint(high=rolling_window_length, device=device, size=())

        # Init noisy cache
        noisy_cache = torch.zeros_like(noises)

        # Init denoising timestep, same across windows
        shared_timesteps = torch.ones(
            [B, rolling_window_length * self.opt.chunk_size],
            device=device,
            dtype=torch.float32,
        )
        for idx, current_timestep in enumerate(reversed(self.denoising_step_list[:-1])):  # from clean to noisy 
            shared_timesteps[:, idx * self.opt.chunk_size:(idx + 1) * self.opt.chunk_size] *= current_timestep

        # Denoising loop with rolling forcing
        for window_index in range(window_num):
            start_chunk = window_start_chunks[window_index]
            end_chunk = window_end_chunks[window_index]  # include

            current_start_frame = start_chunk * self.opt.chunk_size
            current_end_frame = (end_chunk + 1) * self.opt.chunk_size  # not include
            current_num_frames = current_end_frame - current_start_frame

            # `noisy_input`: new noise and previous denoised noisy frames, only last chunk is pure noise
            if current_num_frames == rolling_window_length * self.opt.chunk_size or current_start_frame == 0:
                noisy_input = torch.cat([
                    noisy_cache[:, :, current_start_frame : current_end_frame - self.opt.chunk_size],
                    noises[:, :, current_end_frame - self.opt.chunk_size : current_end_frame]
                ], dim=2)
            else:  # at the end of the video
                noisy_input = noisy_cache[:, :, current_start_frame:current_end_frame].clone()

            if self.opt.input_plucker:
                input_plucker = plucker[:, current_start_frame:current_end_frame, ...]
            else:
                input_plucker = None

            # Init denosing timestep
            if current_num_frames == rolling_window_length * self.opt.chunk_size:
                current_timesteps = shared_timesteps
            elif current_start_frame == 0:
                current_timesteps = shared_timesteps[:, -current_num_frames:]
            elif current_end_frame == f:
                current_timesteps = shared_timesteps[:, :current_num_frames]
            else:
                raise ValueError("`current_num_frames` should be equal to `rolling_window_length` * `self.opt.chunk_size`, or at the first or last window.")

            require_grad = window_index % rolling_window_length == exit_flag

            if current_start_frame == 0 and cond_latents is not None:
                noisy_input = torch.cat([cond_latents, noisy_input[:, :, 1:, ...]], dim=2)
                current_timesteps = torch.cat([torch.zeros_like(current_timesteps[:, :1]), current_timesteps[:, 1:]], dim=1)

            # Diffusion model
            if not require_grad:
                with torch.no_grad():
                    model_outputs = self.diffusion(
                        noisy_input,
                        current_timesteps,
                        prompt_embeds,
                        plucker=input_plucker,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * frame_seqlen,
                        #
                        rolling=True,
                        update_cache=False,
                        chunk_size=self.opt.chunk_size,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=current_start_frame * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                    )
                    model_outputs, da3_outputs = \
                        model_outputs if self.opt.load_da3 else (model_outputs, None)
                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_input, current_timesteps).to(dtype)
            else:
                model_outputs = self.diffusion(
                    noisy_input,
                    current_timesteps,
                    prompt_embeds,
                    plucker=input_plucker,
                    #
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * frame_seqlen,
                    #
                    rolling=True,
                    update_cache=False,
                    chunk_size=self.opt.chunk_size,
                    #
                    kv_cache_da3=self.kv_cache_pos_da3,
                    current_start_da3=current_start_frame * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                )
                model_outputs, da3_outputs = \
                    model_outputs if self.opt.load_da3 else (model_outputs, None)
                pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, noisy_input, current_timesteps).to(dtype)
                outputs[:, :, current_start_frame:current_end_frame] = pred_x0
                if da3_outputs is not None:
                    for da3_k in all_da3_outputs.keys():
                        all_da3_outputs[da3_k][:, current_start_frame:current_end_frame] = da3_outputs[da3_k]

            # Update `noisy_cache`, which is detached from the computation graph
            with torch.no_grad():
                for chunk_idx in range(start_chunk, end_chunk+1):
                    chunk_timestep = current_timesteps[:, 
                                    (chunk_idx - start_chunk)*self.opt.chunk_size : 
                                    (chunk_idx - start_chunk + 1)*self.opt.chunk_size].mean().item()
                    matches = torch.abs(self.denoising_step_list[:-1] - chunk_timestep) < 1e-4
                    chunk_timestep_index = torch.nonzero(matches, as_tuple=True)[0]

                    if chunk_timestep_index == len(self.denoising_step_list[:-1]) - 1:
                        continue

                    next_timestep = self.denoising_step_list[chunk_timestep_index + 1].to(device)

                    noisy_cache[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx+1) * self.opt.chunk_size] = \
                        self.diffusion.scheduler.add_noise(
                            pred_x0.transpose(1, 2).flatten(0, 1),
                            torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                            next_timestep * torch.ones((B * current_num_frames,), device=device, dtype=torch.long)
                        ).unflatten(0, (B, current_num_frames)).transpose(1, 2).to(dtype)\
                        [:, :, (chunk_idx - start_chunk)*self.opt.chunk_size:(chunk_idx - start_chunk+1)*self.opt.chunk_size]

            # Rerun with timestep zero to update the clean cache, which is also detached from the computation graph
            with torch.no_grad():
                context_timesteps = self.opt.context_noise * torch.ones_like(current_timesteps)
                if current_start_frame == 0:
                    context_timesteps = torch.cat([torch.zeros_like(context_timesteps[:, :1]), context_timesteps[:, 1:]], dim=1)
                pred_x0 = self.diffusion.scheduler.add_noise(  # add context noise
                    pred_x0.transpose(1, 2).flatten(0, 1),
                    torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                    context_timesteps.flatten(0, 1),
                ).unflatten(0, (B, current_num_frames)).transpose(1, 2).to(dtype)

                # Only cache the first chunk
                pred_x0 = pred_x0[:, :, :self.opt.chunk_size]
                context_timesteps = context_timesteps[:, :self.opt.chunk_size]
                input_plucker = input_plucker[:, :self.opt.chunk_size]
                self.diffusion(
                    pred_x0,
                    context_timesteps,
                    prompt_embeds,
                    plucker=input_plucker,
                    #
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * frame_seqlen,
                    #
                    rolling=True,
                    update_cache=True,
                    chunk_size=self.opt.chunk_size,
                    #
                    kv_cache_da3=self.kv_cache_pos_da3,
                    current_start_da3=current_start_frame * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                )

        if self.opt.load_da3:
            assert da3_outputs is not None
            da3_outputs = all_da3_outputs

        if self.opt.da3_loss_in_sf:
            assert C2W is not None and fxfycxcy is not None
            H, W = self.opt.input_res

            # Get ground-truth geometry labels
            _, (ray_o, ray_d) = plucker_ray(H//2//self.opt.da3_down_ratio, W//2//self.opt.da3_down_ratio,
                C2W.float(), fxfycxcy.float(), normalize_ray_d=False)
            gt_raymaps = torch.cat([ray_d, ray_o], dim=2).to(dtype)  # (B, f, 6, H/2, W/2)
            gt_pose_enc = torch.cat([
                C2W[:, :, :3, 3].float(),  # (B, f, 3)
                mat_to_quat(C2W[:, :, :3, :3].float()),  # (B, f, 4)
                2. * torch.atan(1. / (2. * fxfycxcy[:, :, 1:2])),  # (B, f, 1); fy -> fov_h
                2. * torch.atan(1. / (2. * fxfycxcy[:, :, 0:1])),  # (B, f, 1); fx -> fov_w
            ], dim=-1).to(dtype)  # (B, f, 9)

            # Compute geometry losses
            ray_loss = self.ray_loss_fn(da3_outputs["ray"], gt_raymaps, confs=da3_outputs["ray_conf"])  # (B, f)
            camera_loss = self.camera_loss_fn(da3_outputs["pose_enc"], gt_pose_enc)  # (B, f)
            da3_outputs["ray_loss"] = ray_loss.mean()
            da3_outputs["camera_loss"] = camera_loss.mean()

        return outputs, da3_outputs

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """
        Initialize a per-GPU KV cache for the Wan model.
        """
        num_blocks = len(self.diffusion.model.blocks)
        num_heads = self.diffusion.model.num_heads
        head_dim = self.diffusion.model.dim // num_heads

        kv_cache_pos = []
        for _ in range(num_blocks):
            kv_cache_pos.append({
                "k": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads, head_dim), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache_pos = kv_cache_pos  # always store the clean cache

        if self.opt.load_da3:
            num_da3_blocks = len(self.diffusion.da3_model.backbone.pretrained.blocks)
            num_heads_da3 = self.diffusion.da3_model.backbone.pretrained.num_heads
            head_dim_da3 = self.diffusion.da3_model.backbone.pretrained.embed_dim // num_heads_da3

            kv_cache_pos_da3 = []
            for _ in range(num_da3_blocks):
                kv_cache_pos_da3.append({
                    "k": torch.zeros((batch_size, num_heads_da3, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, num_heads_da3, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                })
            self.kv_cache_pos_da3 = kv_cache_pos_da3  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """
        Initialize a per-GPU cross-attention cache for the Wan model.
        """
        num_blocks = len(self.diffusion.model.blocks)
        num_heads = self.diffusion.model.num_heads
        head_dim = self.diffusion.model.dim // num_heads

        crossattn_cache_pos = []
        for _ in range(num_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),  # `512` is hard-coded here (max_text_len)
                "v": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
