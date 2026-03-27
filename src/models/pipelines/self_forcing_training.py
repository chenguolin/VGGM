from typing import *
from torch import Tensor
from src.models.wan import WanDiffusionWrapper, WanDiffusionDA3Wrapper
from src.models.modules import VAEDecoderWrapper, TAEHV

import torch
from contextlib import nullcontext
import torch.distributed as dist
import torch.nn.functional as tF

from depth_anything_3.model.utils.transform import mat_to_quat

from src.options import Options
from src.models.modules.decoder_wrapper import ZERO_VAE_CACHE_512, ZERO_VAE_CACHE
from src.models.losses import XYZLoss, CameraLoss
from src.utils import plucker_ray, filter_da3_points, render_pt3d_points, mv_interpolate
from src.utils.distributed import get_sp_world_size


class SelfForcingTrainingPipeline:
    def __init__(self,
        opt: Options,
        diffusion: WanDiffusionWrapper | WanDiffusionDA3Wrapper,
        current_vae_decoder: Optional[VAEDecoderWrapper | TAEHV] = None,
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
        self.ttt_state_pos = None
        self.gdn_state_pos = None

        self.ray_loss_fn, self.camera_loss_fn = XYZLoss(opt), CameraLoss(opt)

        self.current_vae_decoder = current_vae_decoder

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
        #
        clip_latent_lens: Optional[Tensor] = None,  # (B=1, num_clips); for multi-clip generation
    ):
        B, _, f, h, w = noises.shape
        H, W = h * 8, w * 8  # `8`: hard-coded for Wan2.1
        device, dtype = noises.device, noises.dtype

        if cond_latents is not None:
            noises = torch.cat([cond_latents, noises[:, :, 1:, ...]], dim=2)
        output_chunks = []  # collect per-chunk outputs, concatenate at the end to avoid pre-allocating full tensor

        # Set KV cache
        if self.opt.is_causal:
            self._initialize_kv_cache(B, dtype, device)
            self._initialize_crossattn_cache(B, dtype, device)

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
                render_images = torch.zeros((B, self.opt.chunk_size, 3, H, W), dtype=dtype, device=device)  # `8`: hard-coded for Wan2.1
                render_depths = torch.zeros((B, self.opt.chunk_size, H, W), dtype=dtype, device=device)  # `8`: hard-coded for Wan2.1
                render_loss = []
            render_images_vis = render_images

            if self.opt.load_tae:
                vae_cache = None
            else:
                vae_cache = {
                    512: ZERO_VAE_CACHE_512,
                    832: ZERO_VAE_CACHE,
                }[int(W)]
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

        # Temporal denoising loop
        all_da3_outputs, all_points, all_colors, images_f, all_timesteps = [None] * num_chunks, [None] * B, [None] * B, [], []
        for chunk_idx in range(num_chunks):
            this_chunk_latents = noises[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            if self.opt.input_plucker:
                this_chunk_plucker = plucker[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            else:
                this_chunk_plucker = None
            if C2W is not None and fxfycxcy is not None:
                this_chunk_C2W = C2W[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
                this_chunk_fxfycxcy = fxfycxcy[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            else:
                this_chunk_C2W, this_chunk_fxfycxcy = None, None

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
                            extra_condition=input_extra_condition,
                            #
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                            #
                            ttt_state=self.ttt_state_pos,
                            gdn_state=self.gdn_state_pos,
                            #
                            kv_cache_da3=self.kv_cache_pos_da3,
                            current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                            #
                            clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                        )

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
                        extra_condition=input_extra_condition,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        ttt_state=self.ttt_state_pos,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                        #
                        clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    )

                    model_outputs, da3_outputs = \
                        model_outputs if self.opt.load_da3 else (model_outputs, None)

                    pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps).to(dtype)
                    break

            # Record this chunk generated latents
            output_chunks.append(pred_x0)
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
                        extra_condition=input_extra_condition,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        #
                        ttt_state=self.ttt_state_pos,
                        #
                        kv_cache_da3=self.kv_cache_pos_da3,
                        current_start_da3=chunk_idx * self.opt.chunk_size * (frame_seqlen // (self.opt.da3_down_ratio * self.opt.da3_down_ratio) + 1),  # `+1` for camera token
                        #
                        clip_latent_lens=clip_latent_lens,  # for multi-clip generation
                    )

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
                    images_f.append(current_images_f)

                if self.opt.render_loss_in_sf:
                    assert self.opt.load_da3

                    render_masks = input_extra_condition[:, :, -1:, :, :]  # (B, f_chunk, 1, H, W)
                    _render_loss = (tF.mse_loss(current_images_f, render_images,
                        reduction="none") * render_masks).sum() / (render_masks.sum() * 3 + 1e-6)
                    _render_loss = _render_loss[None, None].repeat(B, self.opt.chunk_size)  # (B, f_chunk)
                    render_loss.append(_render_loss)

                # (Optional) Update render images for next chunks
                if chunk_idx < num_chunks - 1:
                    assert self.opt.load_da3

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
                            with torch.no_grad():
                                render_images, render_depths = render_pt3d_points(
                                    H, W, all_points[i], all_colors[i],  # `*8`: hard-coded for Wan2.1
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
            ray_loss = da3_weights * (self.ray_loss_fn(da3_outputs["ray"], gt_raymaps, confs=da3_outputs["ray_conf"])).flatten(0, 1)  # (B*f,)
            camera_loss = da3_weights * (self.camera_loss_fn(da3_outputs["pose_enc"], gt_pose_enc)).flatten(0, 1)  # (B*f,)
            da3_outputs["ray_loss"] = ray_loss.mean()
            da3_outputs["camera_loss"] = camera_loss.mean()
            if self.opt.input_pcrender and self.opt.render_loss_in_sf:
                render_loss = torch.cat(render_loss, dim=1).flatten(0, 1)  # (B*f,)
                da3_outputs["render_loss"] = (da3_weights * render_loss).mean()

        # Concatenate all chunks into the full output tensor
        outputs = torch.cat(output_chunks, dim=2)

        if render_images is not None:
            da3_outputs["images_render"] = render_images_vis

        return outputs, da3_outputs

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

        kv_cache_pos = []
        for _ in range(num_blocks):
            kv_cache_pos.append({
                "k": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads_per_rank, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size, self.opt.max_kvcache_attention_size, num_heads_per_rank, head_dim), dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache_pos = kv_cache_pos  # always store the clean cache

        # TTT state initialization
        if self.opt.use_ttt and self.diffusion.model.use_ttt:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # `init_state` reads TTT fast weight parameters which may be sharded
            # by FSDP; summon full params so we get the correct shapes
            ctx = FSDP.summon_full_params(self.diffusion) \
                if isinstance(self.diffusion, FSDP) else nullcontext()
            with ctx:
                ttt_state_pos = []
                for block in self.diffusion.model.blocks:
                    if hasattr(block.self_attn, "ttt_branch"):
                        ttt_state_pos.append(
                            block.self_attn.ttt_branch.init_state(batch_size, device, dtype))
                    else:
                        ttt_state_pos.append(None)
            self.ttt_state_pos = ttt_state_pos
        else:
            self.ttt_state_pos = None

        # GDN state initialization
        if self.opt.use_gdn and self.diffusion.model.use_gdn:
            gdn_state_pos = []
            for block in self.diffusion.model.blocks:
                if hasattr(block.self_attn, "gdn_branch"):
                    gdn_state_pos.append(
                        block.self_attn.gdn_branch.init_state(batch_size, device, dtype))
                else:
                    gdn_state_pos.append(None)
            self.gdn_state_pos = gdn_state_pos
        else:
            self.gdn_state_pos = None

        if self.opt.load_da3:
            num_da3_blocks = len(self.diffusion.da3_model.backbone.pretrained.blocks)
            num_heads_da3 = self.diffusion.da3_model.backbone.pretrained.num_heads
            head_dim_da3 = self.diffusion.da3_model.backbone.pretrained.embed_dim // num_heads_da3

            # When SP is active, KV cache is stored head-sharded
            num_heads_da3_per_rank = num_heads_da3 // sp_size

            kv_cache_pos_da3 = []
            for _ in range(num_da3_blocks):
                kv_cache_pos_da3.append({
                    "k": torch.zeros((batch_size, num_heads_da3_per_rank, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, num_heads_da3_per_rank, self.opt.da3_max_kvcache_attention_size, head_dim_da3), dtype=dtype, device=device),
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

        # NOTE: `512` is hard-coded here, but we use `text_len` * `num_clips` for cross-attention cache actually
        crossattn_cache_pos = []
        for _ in range(num_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),  # `512` is hard-coded here (max_text_len)
                "v": torch.zeros((batch_size, 512, num_heads, head_dim), dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
