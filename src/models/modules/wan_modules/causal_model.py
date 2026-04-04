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
    rope_apply,
    rope_apply_sp,
)
from src.utils.distributed import get_sp_rank, get_sp_world_size, all_gather, all_to_all, all_split, sync_across_sp_group

# Lazy-compile `flex_attention` on first use so that `use_flexattn=False`
# sessions never pay the compilation cost.
_flex_attention_raw = flex_attention
_flex_attention_compiled = None

def _get_compiled_flex_attention():
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        # wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
        # see https://github.com/pytorch/pytorch/issues/133254
        # change to default for other models
        _flex_attention_compiled = torch.compile(
            _flex_attention_raw, dynamic=False, mode="max-autotune-no-cudagraphs")
        # _flex_attention_compiled = torch.compile(_flex_attention_raw)
    return _flex_attention_compiled


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
    x = _get_compiled_flex_attention()(
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
                 use_flexattn=True,
                 #
                 qk_norm=True,
                 eps=1e-6,
                 #
                 use_ttt=False,
                 ttt_config=None,
                 #
                 use_gdn=False,
                 gdn_config=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        #
        self.sink_size = sink_size
        self.max_attention_size = max_attention_size
        self.rope_outside = rope_outside
        self.use_flexattn = use_flexattn
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

        # (Optional) TTT branch
        self.use_ttt = use_ttt
        if use_ttt:
            from .ttt import TTTBranch
            ttt_config = ttt_config or {}
            self.ttt_branch = TTTBranch(
                dim=dim, num_heads=num_heads, **ttt_config)

        # (Optional) GatedDeltaNet branch
        self.use_gdn = use_gdn
        if use_gdn:
            from .gdn import GDNBranch
            gdn_config = gdn_config or {}
            self.gdn_branch = GDNBranch(
                dim=dim, num_heads=num_heads, **gdn_config)

        # (Optional) Attention gate for progressive SWA -> GDN/TTT transition.
        # Learnable scalar that scales attention output; init to 1. (identity).
        # When trained toward 0., attention is effectively removed for this layer.
        self.use_attn_gate = False  # set to True via `inject_attn_gate()`

    def enable_attn_gate(self):
        """Enable a learnable gate on attention output (init 1.)."""
        self.use_attn_gate = True
        self.attn_gate = nn.Parameter(torch.ones(1))

    def _loop_attention(self, q, k, v, loop_attn_params):
        """Dispatch to diffusion forcing or teacher forcing loop attention."""
        if loop_attn_params["is_teacher_forcing"]:
            return self._loop_attention_teacher_forcing(q, k, v, loop_attn_params)
        else:
            return self._loop_attention_diffusion_forcing(q, k, v, loop_attn_params)

    def _loop_attention_diffusion_forcing(self, q, k, v, params):
        """
        Loop-based windowed attention for diffusion forcing (no teacher forcing).
        Each chunk `i` attends to KV in `[0, (i+1)*block)`, with sink + window limits.
        """
        frame_seqlen = params["frame_seqlen"]
        num_frames = params["num_frames"]
        chunk_size = params["chunk_size"]
        sink_size = params["sink_size"]
        max_attention_size = params["max_attention_size"]

        block = frame_seqlen * chunk_size
        total_len = num_frames * frame_seqlen
        sink_end = sink_size * frame_seqlen
        num_chunks = math.ceil(total_len / block)

        outputs = []
        for i in range(num_chunks):
            q_start = i * block
            q_end = min((i + 1) * block, total_len)
            q_i = q[:, q_start:q_end]

            chunk_end = q_end

            if max_attention_size == -1 or sink_end == 0:
                # No window limit or no sink: take all KV up to `chunk_end`
                if max_attention_size == -1:
                    k_i = k[:, :chunk_end]
                    v_i = v[:, :chunk_end]
                else:
                    # Window limit, no sink
                    window_start = max(0, chunk_end - max_attention_size)
                    k_i = k[:, window_start:chunk_end]
                    v_i = v[:, window_start:chunk_end]
            else:
                # Sink + window
                window = max_attention_size - sink_end
                window_start = max(sink_end, chunk_end - window)
                k_i = torch.cat([k[:, :sink_end], k[:, window_start:chunk_end]], dim=1)
                v_i = torch.cat([v[:, :sink_end], v[:, window_start:chunk_end]], dim=1)

            out_i = attention(q_i, k_i, v_i)
            outputs.append(out_i)

        return torch.cat(outputs, dim=1)

    def _loop_attention_teacher_forcing(self, q, k, v, params):
        """
        Loop-based windowed attention for teacher forcing.
        Sequence layout: `[clean_tokens | noisy_tokens]`, each half has
        `num_frames * frame_seqlen` tokens.

        Clean chunk `i`: same as diffusion forcing, operating within clean half.
        Noisy chunk `i`: attends to context from clean half (sink + window) and
        itself from noisy half.
        """
        frame_seqlen = params["frame_seqlen"]
        num_frames = params["num_frames"]
        chunk_size = params["chunk_size"]
        sink_size = params["sink_size"]
        max_attention_size = params["max_attention_size"]

        block = frame_seqlen * chunk_size
        half_len = num_frames * frame_seqlen
        sink_end = sink_size * frame_seqlen
        num_chunks = math.ceil(half_len / block)

        # Split into clean and noisy halves
        q_clean, q_noisy = q[:, :half_len], q[:, half_len:]
        k_clean, k_noisy = k[:, :half_len], k[:, half_len:]
        v_clean, v_noisy = v[:, :half_len], v[:, half_len:]

        outputs = []

        # --- Clean half: same as diffusion forcing ---
        for i in range(num_chunks):
            q_start = i * block
            q_end = min((i + 1) * block, half_len)
            q_i = q_clean[:, q_start:q_end]

            chunk_end = q_end

            if max_attention_size == -1 or sink_end == 0:
                if max_attention_size == -1:
                    k_i = k_clean[:, :chunk_end]
                    v_i = v_clean[:, :chunk_end]
                else:
                    window_start = max(0, chunk_end - max_attention_size)
                    k_i = k_clean[:, window_start:chunk_end]
                    v_i = v_clean[:, window_start:chunk_end]
            else:
                window = max_attention_size - sink_end
                window_start = max(sink_end, chunk_end - window)
                k_i = torch.cat([k_clean[:, :sink_end], k_clean[:, window_start:chunk_end]], dim=1)
                v_i = torch.cat([v_clean[:, :sink_end], v_clean[:, window_start:chunk_end]], dim=1)

            out_i = attention(q_i, k_i, v_i)
            outputs.append(out_i)

        # --- Noisy half: attend to clean context + noisy self ---
        for i in range(num_chunks):
            q_start = i * block
            q_end = min((i + 1) * block, half_len)
            q_i = q_noisy[:, q_start:q_end]

            # Context from clean half: blocks `[0, i*block)`
            context_end = i * block

            # Self from noisy half
            noisy_k_self = k_noisy[:, q_start:q_end]
            noisy_v_self = v_noisy[:, q_start:q_end]

            if context_end == 0:
                # First noisy chunk: no clean context, only self (no sink per original mask)
                k_i = noisy_k_self
                v_i = noisy_v_self
            else:
                # Gather clean context with sink + window
                if max_attention_size == -1 or sink_end == 0:
                    if max_attention_size == -1:
                        ctx_k = k_clean[:, :context_end]
                        ctx_v = v_clean[:, :context_end]
                    else:
                        # Window budget for clean context = `max_attention_size` - self block size
                        ctx_window = max_attention_size - (sink_size + chunk_size) * frame_seqlen
                        ctx_window_start = max(0, context_end - ctx_window)
                        ctx_k = k_clean[:, ctx_window_start:context_end]
                        ctx_v = v_clean[:, ctx_window_start:context_end]
                else:
                    ctx_window = max_attention_size - (sink_size + chunk_size) * frame_seqlen
                    ctx_window_start = max(sink_end, context_end - ctx_window)
                    # Sink exception: first noisy chunk (`i=0`) doesn't get sink
                    # For `i>0`, include sink tokens
                    ctx_k = torch.cat([k_clean[:, :sink_end], k_clean[:, ctx_window_start:context_end]], dim=1)
                    ctx_v = torch.cat([v_clean[:, :sink_end], v_clean[:, ctx_window_start:context_end]], dim=1)

                # Concatenate clean context + noisy self
                k_i = torch.cat([ctx_k, noisy_k_self], dim=1)
                v_i = torch.cat([ctx_v, noisy_v_self], dim=1)

            out_i = attention(q_i, k_i, v_i)
            outputs.append(out_i)

        return torch.cat(outputs, dim=1)

    def reset_parameters(self):
        # Required by FSDP to materialize meta-device parameters
        # Actual weights are synced from rank 0 via `sync_module_states`
        pass

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
        #
        loop_attn_params=None,
        #
        ttt_state=None,
        #
        gdn_state=None,
        #
        _skip_kv_write=False,  # for gradient checkpointing
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

        # Save input for TTT branch projections
        hidden_states = x

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Save pre-RoPE Q/K/V for TTT / GDN before the KV-cache path may
        # all-to-all them into head-parallel layout
        if self.use_ttt:
            ttt_q, ttt_k, ttt_v = q, k, v
        if self.use_gdn:
            gdn_q, gdn_k, gdn_v = q, k, v

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

            if not self.use_flexattn:
                if sp_size > 1:
                    # Ulysses-style: scatter heads, gather sequence
                    roped_query = all_to_all(roped_query, scatter_dim=2, gather_dim=1)
                    roped_key = all_to_all(roped_key, scatter_dim=2, gather_dim=1)
                    v = all_to_all(v, scatter_dim=2, gather_dim=1)
                x = self._loop_attention(
                    roped_query, roped_key, v, loop_attn_params)
                if sp_size > 1:
                    # Scatter sequence, gather heads
                    x = all_to_all(x, scatter_dim=1, gather_dim=2)
            elif sp_size > 1:
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

                x = _get_compiled_flex_attention()(
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
                if not _skip_kv_write:
                    kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                if not _skip_kv_write:
                    kv_cache["k"][:, local_start_index:local_end_index] = k if self.rope_outside else roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
            # Not exceeding the local attention size
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                if not _skip_kv_write:
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

            if not _skip_kv_write:
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

            if sp_size > 1:
                # Scatter sequence, gather heads
                x = all_to_all(x, scatter_dim=1, gather_dim=2)

        # (Optional) TTT branch (parallel with attention, uses pre-RoPE Q/K/V)
        # Always uses the original seq-parallel Q/K/V saved above; TTT handles
        # its own Ulysses-style all-to-all internally when `sp_size > 1`.

        # Compute grid sizes for short conv (latent patch dims)
        conv_grid = None
        if (self.use_ttt and self.ttt_branch.use_conv) or \
           (self.use_gdn and self.gdn_branch.use_conv):
            _gs = grid_sizes[0]  # assume uniform batch
            conv_grid = (_gs[0].item(), _gs[1].item(), _gs[2].item())

        if self.use_ttt:
            # Detect teacher forcing: sequence has [clean, noisy] layout
            tf_clean_len = None
            if kv_cache is None and (s == seq_lens[0].item() * 2 // sp_size):
                tf_clean_len = seq_lens[0].item()  # total clean tokens (pre-SP)

            # `ttt_state` is mutated in-place (like KV-cache)
            ttt_output = self.ttt_branch(
                ttt_q, ttt_k, ttt_v, hidden_states, ttt_state=ttt_state,
                teacher_forcing_clean_len=tf_clean_len,
                grid_sizes=conv_grid,
            )

        # (Optional) GDN branch (parallel with attention, uses pre-RoPE Q/K/V)
        # Always uses the original seq-parallel Q/K/V saved above; GDN handles
        # its own Ulysses-style all-to-all internally when `sp_size > 1`.
        if self.use_gdn:
            tf_clean_len = None
            if kv_cache is None and (s == seq_lens[0].item() * 2 // sp_size):
                tf_clean_len = seq_lens[0].item()

            # `gdn_state` is mutated in-place (like KV-cache)
            gdn_output = self.gdn_branch(
                gdn_q, gdn_k, gdn_v, hidden_states, gdn_state=gdn_state,
                teacher_forcing_clean_len=tf_clean_len,
                grid_sizes=conv_grid,
            )

        # output
        x = x.flatten(2)
        if self.use_attn_gate:
            x = self.attn_gate * x
        if self.use_ttt:
            x = x + ttt_output
        if self.use_gdn:
            x = x + gdn_output
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
                 use_flexattn=True,
                 #
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 #
                 use_ttt=False,
                 ttt_config=None,
                 #
                 use_gdn=False,
                 gdn_config=None):
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
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, sink_size, max_attention_size, rope_outside, use_flexattn,
            qk_norm, eps, use_ttt=use_ttt, ttt_config=ttt_config,
            use_gdn=use_gdn, gdn_config=gdn_config)
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

    def reset_parameters(self):
        # Required by FSDP to materialize meta-device parameters
        # Actual weights are synced from rank 0 via `sync_module_states`
        pass

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
        loop_attn_params=None,
        #
        ttt_state=None,
        #
        gdn_state=None,
        #
        clip_query_lens=None,
        clip_context_lens=None,
        #
        is_teacher_forcing=False,  # for correct clipwise cross-attention
        #
        _skip_kv_write=False,  # for gradient checkpointing
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
            loop_attn_params=loop_attn_params,
            ttt_state=ttt_state,
            gdn_state=gdn_state,
            _skip_kv_write=_skip_kv_write,
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
            if is_teacher_forcing:
                half = x.shape[1] // 2
                x_clean, x_noisy = x[:, :half], x[:, half:]
                x_clean = x_clean + self.cross_attn(
                    self.norm3(x_clean), context, context_lens,
                    crossattn_cache=crossattn_cache,
                    clip_query_lens=clip_query_lens,
                    clip_context_lens=clip_context_lens,
                )
                x_noisy = x_noisy + self.cross_attn(
                    self.norm3(x_noisy), context, context_lens,
                    crossattn_cache=crossattn_cache,
                    clip_query_lens=clip_query_lens,
                    clip_context_lens=clip_context_lens,
                )
                x = torch.cat([x_clean, x_noisy], dim=1)
            else:
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
                 use_flexattn=True,
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
        self.use_flexattn = use_flexattn
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

        self.use_ttt = False
        self.use_gdn = False

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    #
                                    sink_size, max_attention_size, rope_outside, use_flexattn,
                                    #
                                    qk_norm, cross_attn_norm, eps)
            for i in range(num_layers)
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
        ttt_state: list = None,
        #
        gdn_state: list = None,
        #
        clip_query_lens: Optional[int] = None,
        clip_context_lens: Optional[int] = None,
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

        for block_index, block in enumerate(self.blocks):
            block_ttt_state = ttt_state[block_index] if ttt_state is not None else None
            block_gdn_state = gdn_state[block_index] if gdn_state is not None else None
            this_kv_cache = kv_cache[block_index]

            if torch.is_grad_enabled() and (self.use_gradient_checkpointing_offload or self.use_gradient_checkpointing):
                # Snapshot the two scalar indices so that the recompute wrapper
                # can restore them before re-running the block.  K/V tensor
                # contents are NOT cloned — we pass `_skip_kv_write=True` during
                # recompute so the block only *reads* the cache (to produce the
                # correct attention output) without mutating K/V storage.
                saved_global_end = this_kv_cache["global_end_index"].clone()
                saved_local_end = this_kv_cache["local_end_index"].clone()
                this_crossattn_cache = crossattn_cache[block_index] if crossattn_cache is not None else None
                saved_crossattn_is_init = this_crossattn_cache["is_init"] if this_crossattn_cache is not None else None

                def create_kv_restoring_forward(module, kvc, s_global, s_local, cac, s_crossattn_is_init):
                    first_call = [True]  # forward pass, not recompute

                    def custom_forward(*inputs, **kwargs):
                        if first_call[0]:
                            first_call[0] = False
                            return module(*inputs, **kwargs)
                        # Recompute: restore indices and skip KV cache writes
                        kvc["global_end_index"].copy_(s_global)
                        kvc["local_end_index"].copy_(s_local)
                        if cac is not None:
                            cac["is_init"] = s_crossattn_is_init
                        kwargs["_skip_kv_write"] = True
                        return module(*inputs, **kwargs)
                    return custom_forward

                kwargs.update({
                    "kv_cache": this_kv_cache,
                    "crossattn_cache": this_crossattn_cache,
                    "current_start": current_start,
                    "ttt_state": block_ttt_state,
                    "gdn_state": block_gdn_state,
                })
                ckpt_fn = create_kv_restoring_forward(
                    block, this_kv_cache, saved_global_end, saved_local_end,
                    this_crossattn_cache, saved_crossattn_is_init)
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            ckpt_fn, x, **kwargs, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        ckpt_fn, x, **kwargs, use_reentrant=False)
            else:
                kwargs.update({
                    "kv_cache": this_kv_cache,
                    "crossattn_cache": crossattn_cache[block_index] if crossattn_cache is not None else None,
                    "current_start": current_start,
                    "ttt_state": block_ttt_state,
                    "gdn_state": block_gdn_state,
                })
                x = block(x, **kwargs)

        # Sequence parallelism: gather sequences before head
        if sp_size > 1:
            x = all_gather(x, dim=1)

        # head & unpatchify
        x = self.head(x, e.unflatten(0, (bt, seq_len)))
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]

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

        # Construct blockwise causal attn mask
        frame_seqlen = h * w // (self.patch_size[1] * self.patch_size[2])
        if self.use_flexattn:
            if self.block_mask is None:
                if clean_x is not None:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=f,
                        frame_seqlen=frame_seqlen,
                        sink_size=self.sink_size,
                        chunk_size=self.chunk_size,
                        max_attention_size=self.max_attention_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=f,
                        frame_seqlen=frame_seqlen,
                        sink_size=self.sink_size,
                        chunk_size=self.chunk_size,
                        max_attention_size=self.max_attention_size,
                    )
        else:
            # Loop-based attention: pass parameters instead of block mask
            self.block_mask = None

        # Sequence parallelism: chunk sequences across ranks
        sp_size = get_sp_world_size()
        if sp_size > 1:
            assert x.size(1) % sp_size == 0
            x = all_split(sync_across_sp_group(x), dim=1)
            e0 = all_split(sync_across_sp_group(e0), dim=1)

        # arguments
        loop_attn_params = None
        if not self.use_flexattn:
            loop_attn_params = dict(
                frame_seqlen=frame_seqlen,
                num_frames=f,
                chunk_size=self.chunk_size,
                sink_size=self.sink_size,
                max_attention_size=self.max_attention_size,
                is_teacher_forcing=clean_x is not None,
            )
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            #
            block_mask=self.block_mask,
            loop_attn_params=loop_attn_params,
            #
            clip_query_lens=clip_query_lens,
            clip_context_lens=clip_context_lens,
            #
            is_teacher_forcing=clean_x is not None,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

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

        # Sequence parallelism: gather sequences before head
        if sp_size > 1:
            x = all_gather(x, dim=1)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head & unpatchify
        x = self.head(x, e.unflatten(0, (bt, seq_len)))
        x = self.unpatchify(x, grid_sizes)

        return [u.float() for u in x]

    def reset_parameters(self):
        # Required by FSDP to materialize meta-device parameters
        # Actual weights are synced from rank 0 via `sync_module_states`
        pass

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            kwargs.pop("ttt_state", None)  # not used in training forward
            kwargs.pop("gdn_state", None)  # not used in training forward
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

    def inject_ttt(self, ttt_layers=None, ttt_config=None):
        """
        Inject TTTBranch into specified attention blocks AFTER pretrained weights
        have been loaded. This avoids TTT parameters interfering with
        `from_pretrained` / `load_state_dict`.

        Args:
            ttt_layers: List of layer indices to inject TTT into. If None and
                `ttt_config` is provided, inject into all layers.
            ttt_config: Dict of TTTBranch kwargs. If None, no injection is done.
        """
        if ttt_config is None and ttt_layers is None:
            return

        # Determine which layers get TTT
        if ttt_layers is None and ttt_config is not None:
            ttt_layer_set = set(range(self.num_layers))
        elif ttt_layers is not None:
            ttt_layer_set = set(ttt_layers)
        else:
            ttt_layer_set = set()

        if len(ttt_layer_set) == 0:
            return

        self.use_ttt = True
        ttt_config = ttt_config or {}

        from .ttt import TTTBranch
        for i in ttt_layer_set:
            block = self.blocks[i]
            sa = block.self_attn
            sa.use_ttt = True
            sa.ttt_branch = TTTBranch(
                dim=sa.dim, num_heads=sa.num_heads, **ttt_config)

    def inject_gdn(self, gdn_layers=None, gdn_config=None):
        """
        Inject GDNBranch into specified attention blocks AFTER pretrained weights
        have been loaded. This avoids GDN parameters interfering with
        `from_pretrained` / `load_state_dict`.

        Args:
            gdn_layers: List of layer indices to inject GDN into. If None and
                `gdn_config` is provided, inject into all layers.
            gdn_config: Dict of GDNBranch kwargs. If None, no injection is done.
        """
        if gdn_config is None and gdn_layers is None:
            return

        # Determine which layers get GDN
        if gdn_layers is None and gdn_config is not None:
            gdn_layer_set = set(range(self.num_layers))
        elif gdn_layers is not None:
            gdn_layer_set = set(gdn_layers)
        else:
            gdn_layer_set = set()

        if len(gdn_layer_set) == 0:
            return

        self.use_gdn = True
        gdn_config = gdn_config or {}

        from .gdn import GDNBranch
        for i in gdn_layer_set:
            block = self.blocks[i]
            sa = block.self_attn
            sa.use_gdn = True
            sa.gdn_branch = GDNBranch(
                dim=sa.dim, num_heads=sa.num_heads, **gdn_config)

    def inject_attn_gate(self, attn_gate_layers=None):
        """
        Inject learnable attention gates into specified layers AFTER pretrained
        weights have been loaded. Each gate is a scalar parameter initialized
        to 1. that multiplies the attention output before adding TTT/GDN output.

        Used for progressive SWA -> GDN/TTT transition: train the gate toward 0
        to let GDN/TTT take over, then remove attention computation entirely.

        Args:
            attn_gate_layers: List of layer indices. If None, no injection.
        """
        if attn_gate_layers is None:
            return

        for i in attn_gate_layers:
            self.blocks[i].self_attn.enable_attn_gate()
