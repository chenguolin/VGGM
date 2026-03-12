from typing import *
from torch import Tensor

import math
import torch
from torch import nn

from .wan_modules.model import Head, WanAttentionBlock, rope_params
from src.utils.distributed import get_sp_world_size, all_gather, all_split, sync_across_sp_group


class DDT(nn.Module):
    """Decoupled Diffusion Transformer.

    Modified from `WanModel` in `wan_modules.model`.
    """
    def __init__(self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        dim=2048,
        ffn_dim=8192,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        #
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        #
        ddt_fusion=False,
    ):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        #
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # fusion
        self.ddt_fusion = ddt_fusion
        if ddt_fusion:
            self.fusion_layer = nn.Linear(2 * dim, dim)

        # embeddings
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                              #
                              qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.d = d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False

    def forward(self,
        x,
        v,
        e,
        context,
        grid_sizes,
        seq_lens,
        #
        clip_query_lens: Optional[Tensor] = None,
        clip_context_lens: Optional[Tensor] = None,
    ):
        # params
        device = x.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # fusion
        if self.ddt_fusion:
            x = self.fusion_layer(torch.cat([x, v], dim=-1))

        # time embeddings
        bt = x.size(0)
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, (bt, seq_lens.max()))

        # context
        context_lens = None

        # Sequence parallelism: chunk sequences across ranks
        sp_size = get_sp_world_size()
        if sp_size > 1:
            assert x.size(1) % sp_size == 0
            x = all_split(sync_across_sp_group(x), dim=1)
            e0 = all_split(sync_across_sp_group(e0), dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            #
            clip_query_lens=clip_query_lens,
            clip_context_lens=clip_context_lens,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:

            if self.training and self.use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        **kwargs,
                        use_reentrant=False,
                    )
            elif self.training and self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # Sequence parallelism: gather sequences before head
        if sp_size > 1:
            x = all_gather(x, dim=1)

        # head & unpatchify
        x = self.head(x, e.unflatten(0, (bt, seq_lens.max())))
        x = self.unpatchify(x, grid_sizes)  # (B, N, D) -> (B, C, f, h, w)

        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    @torch.no_grad()
    def init_weights(self):
        r"""
        Initialize model parameters using Xavier or zero initialization.

        Zero-init and identity-init strategy ensures DDT is identity at initialization:
            - `fusion_layer` identity-init -> `fusion_layer(cat([x, v])) = v` -> `x = v`
            - `time_projection` zero -> `e0 = 0` -> `e[2]` and `e[5]` gate self-attn/FFN to zero
            - `block.modulation` zero -> same gating effect
            - `cross_attn.o` zero -> cross-attention output is zero (not gated by modulation)
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # fusion layer: identity-init
        if self.ddt_fusion:
            nn.init.zeros_(self.fusion_layer.weight)
            nn.init.zeros_(self.fusion_layer.bias)
            self.fusion_layer.weight[:, self.dim:].copy_(torch.eye(self.dim))

        # time modulation: zero-init
        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for block in self.blocks:
            nn.init.zeros_(block.modulation)

        # cross-attention output projection: zero-init
        for block in self.blocks:
            nn.init.zeros_(block.cross_attn.o.weight)
            if block.cross_attn.o.bias is not None:
                nn.init.zeros_(block.cross_attn.o.bias)
