from typing import *
from torch import Tensor
from src.models.wan import WanDiffusionWrapper, WanDiffusionDA3Wrapper

import torch
import torch.distributed as dist

from src.options import Options


class SelfForcingTrainingPipeline:
    def __init__(self,
        opt: Options,
        diffusion: WanDiffusionWrapper | WanDiffusionDA3Wrapper,
    ):
        super().__init__()

        self.opt = opt
        self.diffusion = diffusion
        self.diffusion.scheduler.set_timesteps(self.opt.num_train_timesteps, training=True)

        self.denoising_step_list = torch.tensor(opt.denoising_step_list, dtype=torch.long)
        if opt.warp_denoising_step:
            timesteps = torch.cat((diffusion.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[self.opt.num_train_timesteps - self.denoising_step_list]

        self.kv_cache_pos = None
        self.crossattn_cache_pos = None
        self.kv_cache_pos_da3 = None

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

        # Auto-regression steps
        assert f % self.opt.chunk_size == 0
        num_chunks = f // self.opt.chunk_size
        frame_seqlen = h * w // 4  # `4`: hard-coded for 2x2 patch embedding in DiT
        exit_flags = self.generate_and_sync_list(num_chunks, len(self.denoising_step_list), device)

        # Temporal denoising loop
        for chunk_idx in range(num_chunks):
            this_chunk_latents = noises[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]
            if self.opt.input_plucker:
                this_chunk_plucker = plucker[:, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...]

            # Spatial denoising loop
            for i, timestep in enumerate(self.denoising_step_list):
                # Only backprop at the randomly selected timestep (consistent across all ranks)
                if self.opt.same_step_across_chunks:
                    exit_flag = (i == exit_flags[0])
                else:
                    exit_flag = (i == exit_flags[chunk_idx])

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
                            #
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                        )
                        if self.opt.load_da3:
                            model_outputs, da3_outputs = model_outputs

                        if self.opt.deterministic_inference:
                            this_chunk_latents = self.diffusion.scheduler.step(
                                model_outputs.transpose(1, 2).flatten(0, 1),
                                timesteps.flatten(0, 1),
                                this_chunk_latents.transpose(1, 2).flatten(0, 1),
                            ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
                        else:
                            pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps)
                            next_timesteps = self.denoising_step_list[i + 1] * torch.ones_like(timesteps)
                            if chunk_idx == 0 and cond_latents is not None:
                                next_timesteps = torch.cat([torch.zeros_like(next_timesteps[:, :1]), next_timesteps[:, 1:]], dim=1)

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
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                    )
                    if self.opt.load_da3:
                        model_outputs, da3_outputs = model_outputs

                    if self.opt.deterministic_inference:
                        this_chunk_latents = self.diffusion.scheduler.step(
                            model_outputs.transpose(1, 2).flatten(0, 1),
                            timesteps.flatten(0, 1),
                            this_chunk_latents.transpose(1, 2).flatten(0, 1),
                        ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2)  # (B, D, f_chunk, h, w)
                    else:
                        pred_x0 = self.diffusion._convert_flow_pred_to_x0(model_outputs, this_chunk_latents, timesteps)
                    break

            # Record this chunk generated latents
            outputs[:, :, chunk_idx * self.opt.chunk_size:(chunk_idx + 1) * self.opt.chunk_size, ...] = pred_x0

            # Rerun with timestep `context_timestep` to update KV cache
            if self.opt.is_causal:
                context_timesteps = self.opt.context_noise * torch.ones_like(timesteps)
                if chunk_idx == 0 and cond_latents is not None:
                    context_timesteps = torch.cat([torch.zeros_like(context_timesteps[:, :1]), context_timesteps[:, 1:]], dim=1)

                pred_x0 = self.diffusion.scheduler.add_noise(  # add context noise
                    pred_x0.transpose(1, 2).flatten(0, 1),
                    torch.randn_like(pred_x0.transpose(1, 2).flatten(0, 1)),
                    context_timesteps.flatten(0, 1),
                ).unflatten(0, (B, self.opt.chunk_size)).transpose(1, 2).to(dtype)
                with torch.no_grad():
                    self.diffusion(
                        pred_x0,
                        context_timesteps,
                        prompt_embeds,
                        plucker=this_chunk_plucker,
                        #
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=chunk_idx * self.opt.chunk_size * frame_seqlen,
                    )

        # # Return the denoised timesteps
        # if not self.opt.same_step_across_chunks:
        #     denoised_timestep_from, denoised_timestep_to = None, None
        # elif exit_flags[0] == len(self.denoising_step_list) - 1:
        #     denoised_timestep_to = 0
        #     denoised_timestep_from = 1000 - torch.argmin(
        #         (self.diffusion.scheduler.timesteps - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()
        # else:
        #     denoised_timestep_to = 1000 - torch.argmin(
        #         (self.diffusion.scheduler.timesteps - self.denoising_step_list[exit_flags[0] + 1]).abs(), dim=0).item()
        #     denoised_timestep_from = 1000 - torch.argmin(
        #         (self.diffusion.scheduler.timesteps - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()

        return outputs

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
