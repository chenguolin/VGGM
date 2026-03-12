# Referred from https://github.com/guandeh17/Self-Forcing/blob/main/wan/modules/causal_model.py

from typing import *
from torch import Tensor

from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .attention import attention
from .model import (
    WanRMSNorm,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    MLPProj,
    ModelMixin, ConfigMixin,
    register_to_config,
    sinusoidal_embedding_1d,
    rope_params,
    get_1d_rotary_pos_embed_riflex,
    rope_apply,
    rope_apply_sp,
)
from src.utils.distributed import get_sp_rank, get_sp_world_size, all_gather, all_to_all, all_split, sync_across_sp_group

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
# flex_attention = torch.compile(flex_attention)


@torch.amp.autocast('cuda', enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


# ============================================================================
# Sequence Parallelism Support
# ============================================================================

@torch.amp.autocast('cuda', enabled=False)
def rope_apply_sp_tf(x, grid_sizes, freqs):
    """
    Apply RoPE with sequence parallelism support for teacher forcing.

    Args:
        x: [B, L//sp, N, C]
        grid_sizes: [B, 3] containing (F, H, W)
        freqs: [M, C//2]
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # for teacher forcing, RoPE applied to clean and noisy parts are the same
        freqs_i = torch.cat([freqs_i, freqs_i], dim=0)

        # apply rotary embedding with SP offset
        sp_rank = get_sp_rank()
        freqs_i_rank = freqs_i[(sp_rank * s):((sp_rank + 1) * s), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def distributed_flex_attention(
    q,
    k,
    v,
    block_mask,
):
    """
    Ulysses-style distributed flex attention using all-to-all communication.

    Args:
        q: [B, Lq // sp, Nq, C1]
        k: [B, Lk // sp, Nk, C1]
        v: [B, Lk // sp, Nk, C2]

    Returns:
        x: [B, Lq // sp, Nq, C2]
    """
    # Scatter heads, gather sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1)  # [B, Lq, Nq//sp, C1]
    k = all_to_all(k, scatter_dim=2, gather_dim=1)  # [B, Lk, Nk//sp, C1]
    v = all_to_all(v, scatter_dim=2, gather_dim=1)  # [B, Lk, Nk//sp, C2]

    # Padding for flexattention
    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]  # `128`: block size for flexattention; TODO: make it configurable
    if padded_length > 0:
        q = torch.cat([q, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=q.dtype)], dim=1)
        k = torch.cat([k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=k.dtype)], dim=1)
        v = torch.cat([v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)

    # Apply flex attention on full sequence with split heads
    x = flex_attention(
        query=q.transpose(2, 1),
        key=k.transpose(2, 1),
        value=v.transpose(2, 1),
        block_mask=block_mask,
    ).transpose(2, 1)
    if padded_length > 0:
        x = x[:, :-padded_length, ...]

    # Scatter sequence, gather heads
    x = all_to_all(x, scatter_dim=1, gather_dim=2)  # [B, Lq//sp, Nq, C2]
    return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 #
                 sink_size=0,
                 max_attention_size=32760,  # 21 x 480 x 832 -> 21 x 30 x 52
                 rope_outside=False,
                 #
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        #
        self.sink_size = sink_size
        self.max_attention_size = max_attention_size
        self.rope_outside = rope_outside
        #
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        #
        block_mask=None,
        kv_cache=None,
        current_start=0,  # use with `kv_cache`
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        sp_size = get_sp_world_size()

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # No KV cache
        if kv_cache is None:
            # Teacher forcing training
            if (s == seq_lens[0].item() * 2 // sp_size):
                if sp_size > 1:
                    roped_query = rope_apply_sp_tf(q, grid_sizes, freqs).type_as(v)
                    roped_key = rope_apply_sp_tf(k, grid_sizes, freqs).type_as(v)
                else:
                    q_chunk = torch.chunk(q, 2, dim=1)
                    k_chunk = torch.chunk(k, 2, dim=1)
                    roped_query = []
                    roped_key = []
                    # RoPE should be same for clean and noisy parts
                    for ii in range(2):
                        rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                        rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                        roped_query.append(rq)
                        roped_key.append(rk)

                    roped_query = torch.cat(roped_query, dim=1)
                    roped_key = torch.cat(roped_key, dim=1)

            # Not teacher forcing training
            else:
                if sp_size > 1:
                    roped_query = rope_apply_sp(q, grid_sizes, freqs).type_as(v)
                    roped_key = rope_apply_sp(k, grid_sizes, freqs).type_as(v)
                else:
                    roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                    roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            if sp_size > 1:
                x = distributed_flex_attention(
                    roped_query,
                    roped_key,
                    v,
                    block_mask,
                )
            else:
                # Padding for flexattention
                padded_length = math.ceil(roped_query.shape[1] / 128) * 128 - roped_query.shape[1]  # `128`: block size for flexattention; TODO: make it configurable
                if padded_length > 0:
                    roped_query = torch.cat([roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)], dim=1)
                    roped_key = torch.cat([roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)], dim=1)
                    v = torch.cat([v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)

                x = flex_attention(
                    query=roped_query.transpose(2, 1),
                    key=roped_key.transpose(2, 1),
                    value=v.transpose(2, 1),
                    block_mask=block_mask,
                ).transpose(2, 1)
                if padded_length > 0:
                    x = x[:, :-padded_length, ...]

        # Use KV cache
        else:
            if sp_size > 1:
                # Scatter heads, gather sequence; KV cache is stored head-sharded
                q = all_to_all(q, scatter_dim=2, gather_dim=1)
                k = all_to_all(k, scatter_dim=2, gather_dim=1)
                v = all_to_all(v, scatter_dim=2, gather_dim=1)

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            if not self.rope_outside:
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]

            # If the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            if (current_end > kv_cache["global_end_index"].item()) and \
                (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k if self.rope_outside else roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            # Not exceeding the local attention size
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k if self.rope_outside else roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

            if sink_tokens > 0:
                input_k = torch.cat([
                    kv_cache["k"][:, :sink_tokens],
                    kv_cache["k"][:, max(sink_tokens, local_end_index - self.max_attention_size + sink_tokens):local_end_index],
                ], dim=1)
                input_v = torch.cat([
                    kv_cache["v"][:, :sink_tokens],
                    kv_cache["v"][:, max(sink_tokens, local_end_index - self.max_attention_size + sink_tokens):local_end_index],
                ], dim=1)
            else:
                input_k = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
                input_v = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]

            # (Optional) Apply RoPE here, instead of in KV cache
            if self.rope_outside:
                assert q.shape[1] // frame_seqlen == grid_sizes[0, 0]
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs,
                    start_frame=(local_end_index - max(0, local_end_index - self.max_attention_size) - q.shape[1]) // frame_seqlen,
                ).type_as(v)

                grid_sizes_kv = deepcopy(grid_sizes)
                grid_sizes_kv[:, 0] = (local_end_index - max(0, local_end_index - self.max_attention_size)) // frame_seqlen
                assert input_k.shape[1] // frame_seqlen == grid_sizes_kv[0, 0]
                input_k = causal_rope_apply(
                    input_k, grid_sizes_kv, freqs, start_frame=0).type_as(v)

            x = attention(roped_query, input_k, input_v)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

            if sp_size > 1:
                # Scatter sequence, gather heads
                x = all_to_all(x, scatter_dim=1, gather_dim=2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 #
                 sink_size=0,
                 max_attention_size=32760,  # 21 x 480 x 832 -> 21 x 30 x 52
                 rope_outside=False,
                 #
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        #
        self.sink_size = sink_size
        self.max_attention_size = max_attention_size
        self.rope_outside = rope_outside
        #
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, sink_size, max_attention_size, rope_outside, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        #
        block_mask=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        #
        clip_query_lens=None,
        clip_context_lens=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [_e.squeeze(2) for _e in e]
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            seq_lens, grid_sizes, freqs,
            block_mask, kv_cache, current_start,
        )
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(
            x,
            context,
            context_lens,
            e,
            crossattn_cache=None,
            clip_query_lens=None,
            clip_context_lens=None,
        ):
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens,
                crossattn_cache=crossattn_cache,
                clip_query_lens=clip_query_lens,
                clip_context_lens=clip_context_lens,
            )
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(
            x, context, context_lens, e, crossattn_cache,
            clip_query_lens=clip_query_lens,
            clip_context_lens=clip_context_lens,
        )

        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 #
                 sink_size=0,
                 chunk_size=1,
                 max_attention_size=32760,  # 21 x 480 x 832 -> 21 x 30 x 52
                 rope_outside=False,
                 #
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        #
        self.sink_size = sink_size
        self.chunk_size = chunk_size
        self.max_attention_size = max_attention_size
        self.rope_outside = rope_outside
        #
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    #
                                    sink_size, max_attention_size, rope_outside,
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

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False

        self.block_mask = None

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int = 21,  # `21`: 81
        frame_seqlen: int = 1560,  # `1560`: 480 x 832
        sink_size=0,
        chunk_size=1,
        max_attention_size=-1,
    ) -> BlockMask:
        """
        We use flexattention to construct the block-wise causal attention mask.
        """
        total_length = num_frames * frame_seqlen * 2

        # We do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length  # `128`: block size for flexattention; TODO: make it configurable

        clean_ends = num_frames * frame_seqlen
        # For clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # For noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * chunk_size
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # Attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # Attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        if max_attention_size != -1:
            max_attention_size_clean = max_attention_size - sink_size * frame_seqlen  # exclude the sink
            max_attention_size_noise = max_attention_size - (sink_size + chunk_size) * frame_seqlen  # exclude the sink and self

        def attention_mask(b, h, q_idx, kv_idx):
            # First design the mask for clean frames
            if max_attention_size == -1:
                clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            else:
                clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx]) & (kv_idx >= context_ends[q_idx] - max_attention_size_clean)
            # Then design the mask for noisy frames
            # Noisy frames will attend to all clean preceeding clean frames + itself
            if max_attention_size == -1:
                C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
                C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            else:
                C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
                C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx]) & (kv_idx >= noise_context_ends[q_idx] - max_attention_size_noise)
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            sink_mask = (kv_idx <= sink_size * frame_seqlen) & ((q_idx < clean_ends) | (q_idx > clean_ends + chunk_size * frame_seqlen))
            return eye_mask | clean_mask | noise_mask | sink_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
            BLOCK_SIZE=128,  # `128`: block size for flexattention; TODO: make it configurable
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\nCache a chunk-wise teacher forcing causal mask with chunk size of [{chunk_size}] frames")
            print(block_mask)

        # Visualization for debug
        import imageio
        import numpy as np
        from torch.nn.attention.flex_attention import create_mask

        mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                           padded_length, KV_LEN=total_length + padded_length, device=device)
        import cv2
        mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        imageio.imwrite("temp_causal_mask.jpg", np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,  # `21`: 81
        frame_seqlen: int = 1560,  # `1560`: 480 x 832
        sink_size=0,
        chunk_size=1,
        max_attention_size=-1,
    ) -> BlockMask:
        """
        We use flexattention to construct the block-wise causal attention mask.
        """
        total_length = num_frames * frame_seqlen

        # We do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length  # `128`: block size for flexattention; TODO: make it configurable

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * chunk_size,
            device=device
        )

        for idx in frame_indices:
            ends[idx:idx + frame_seqlen * chunk_size] = idx + frame_seqlen * chunk_size

        if max_attention_size != -1:
            max_attention_size = max_attention_size - sink_size * frame_seqlen

        def attention_mask(b, h, q_idx, kv_idx):
            sink_mask = kv_idx <= sink_size * frame_seqlen

            if max_attention_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx) | sink_mask
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - max_attention_size))) | (q_idx == kv_idx) | sink_mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
            BLOCK_SIZE=128,  # `128`: block size for flexattention; TODO: make it configurable
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\nCache a chunk-wise diffusion forcing causal mask with chunk size of [{chunk_size}] frames\n")
            print(block_mask)

        # Visualization for debug
        import imageio
        import numpy as np
        from torch.nn.attention.flex_attention import create_mask

        mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                           padded_length, KV_LEN=total_length + padded_length, device=device)
        import cv2
        mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        imageio.imwrite("temp_causal_mask.jpg", np.uint8(255. * mask))

        return block_mask

    def enable_riflex(
        self,
        k=6,
        L_test=66,
        L_test_scale=4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
            ],
            dim=1
        ).to(device)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        add_embeds=None,
        #
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        #
        clip_query_lens: Optional[int] = None,
        clip_context_lens: Optional[int] = None,
        #
        return_feat_layer_idx: Optional[int] = None,
        not_head_and_unpatchify: bool = False,
        return_ddt_inputs: bool = False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times, if `self.chunk_size` is 1,
        otherwise, it will be run for num_frame // chunk_size times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            add_embeds (List[Tensor], *optional*):
                List of add embeddings for video inputs with shape [D, F, H', W'], same shape as x after patch embedding

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        if add_embeds is not None:
            x = [u + v for u, v in zip(x, add_embeds)]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if seq_len is None:
            seq_len = seq_lens.max()
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, (bt, seq_len))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # context = self.text_embedding(
        #     torch.stack([
        #         torch.cat(
        #             [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        #         for u in context
        #     ]))
        context = self.text_embedding(torch.stack(context))  # (B, L*num_clips, D')

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        ddt_inputs = dict(
            x=x.clone(),
            e=e,
            context=context,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
        )

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

        inter_feats = None
        for block_index, block in enumerate(self.blocks):

            if torch.is_grad_enabled() and self.use_gradient_checkpointing_offload:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                    }
                )
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, **kwargs,
                        use_reentrant=False,
                    )
            elif torch.is_grad_enabled() and self.use_gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                    }
                )
                x = block(x, **kwargs)

            if return_feat_layer_idx is not None and (
                block_index == return_feat_layer_idx or
                block_index == int(return_feat_layer_idx * len(self.blocks))
            ):
                inter_feats = x.copy()

        # Sequence parallelism: gather sequences before head
        if sp_size > 1:
            x = all_gather(x, dim=1)
            if inter_feats is not None:
                inter_feats = all_gather(inter_feats, dim=1)

        # head & unpatchify
        if not not_head_and_unpatchify:  # (B, N, D) -> (B, C, f, h, w)
            x = self.head(x, e.unflatten(0, (bt, seq_len)))
            x = self.unpatchify(x, grid_sizes)

        if return_ddt_inputs:
            if inter_feats is None:
                return [u.float() for u in x], ddt_inputs
            else:
                return [u.float() for u in x], [v.float() for v in inter_feats], ddt_inputs
        else:
            if inter_feats is None:
                return [u.float() for u in x]
            else:
                return [u.float() for u in x], [v.float() for v in inter_feats]

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        add_embeds=None,
        #
        clean_x: Optional[Tensor] = None,
        aug_t: Optional[Tensor] = None,
        #
        clip_query_lens: Optional[Tensor] = None,
        clip_context_lens: Optional[Tensor] = None,
        #
        return_feat_layer_idx: Optional[int] = None,
        not_head_and_unpatchify: bool = False,
        return_ddt_inputs: bool = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B] or [B, L=F*H*W]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            add_embeds (List[Tensor], *optional*):
                List of add embeddings for video inputs with shape [D, F, H', W'], same shape as x after patch embedding

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        f, h, w = x[0].shape[1:]  # assume all inputs have the same shape

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        if add_embeds is not None:
            x = [u + v for u, v in zip(x, add_embeds)]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if seq_len is None:
            seq_len = seq_lens.max()
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        # with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, (bt, seq_len))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # context = self.text_embedding(
        #     torch.stack([
        #         torch.cat(
        #             [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        #         for u in context
        #     ]))
        context = self.text_embedding(torch.stack(context))  # (B, L*num_clips, D')

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Clean inputs for teacher forcing
        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            if add_embeds is not None:
                clean_x = [u + v for u, v in zip(clean_x, add_embeds)]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]
            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean.max() - u.size(1), u.size(2))],
                          dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(0, (bt, seq_lens_clean.max()))
            # assert e_clean.dtype == torch.float32 and e0_clean.dtype == torch.float32
            e0 = torch.cat([e0_clean, e0], dim=1)
            if clip_query_lens is not None:
                clip_query_lens = torch.cat([clip_query_lens, clip_query_lens], dim=1)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                self.block_mask = self._prepare_teacher_forcing_mask(
                    device,
                    num_frames=f,
                    frame_seqlen=h * w // (self.patch_size[1] * self.patch_size[2]),
                    sink_size=self.sink_size,
                    chunk_size=self.chunk_size,
                    max_attention_size=self.max_attention_size,
                )
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device,
                    num_frames=f,
                    frame_seqlen=h * w // (self.patch_size[1] * self.patch_size[2]),
                    sink_size=self.sink_size,
                    chunk_size=self.chunk_size,
                    max_attention_size=self.max_attention_size,
                )

        ddt_inputs = dict(
            x=x.clone(),
            e=torch.cat([e_clean, e], dim=1) \
                if clean_x is not None else e,
            context=context,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
            block_mask=self.block_mask,
        )

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
            block_mask=self.block_mask,
            #
            clip_query_lens=clip_query_lens,
            clip_context_lens=clip_context_lens,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        inter_feats = None
        for block_index, block in enumerate(self.blocks):

            if self.training and self.use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, **kwargs,
                        use_reentrant=False,
                    )
            elif self.training and self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if return_feat_layer_idx is not None and (
                block_index == return_feat_layer_idx or
                block_index == int(return_feat_layer_idx * len(self.blocks))
            ):
                inter_feats = x.copy()

        # Sequence parallelism: gather sequences before head
        if sp_size > 1:
            x = all_gather(x, dim=1)
            if inter_feats is not None:
                inter_feats = all_gather(inter_feats, dim=1)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]
            if inter_feats is not None:
                inter_feats = inter_feats[:, inter_feats.shape[1] // 2:]

        # head & unpatchify
        if not not_head_and_unpatchify:  # (B, N, D) -> (B, C, f, h, w)
            x = self.head(x, e.unflatten(0, (bt, seq_len)))
            x = self.unpatchify(x, grid_sizes)

        if return_ddt_inputs:
            if inter_feats is None:
                return [u.float() for u in x], ddt_inputs
            else:
                return [u.float() for u in x], [v.float() for v in inter_feats], ddt_inputs
        else:
            if inter_feats is None:
                return [u.float() for u in x]
            else:
                return [u.float() for u in x], [v.float() for v in inter_feats]

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

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

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
