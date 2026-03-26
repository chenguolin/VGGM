# GatedDeltaNet (GDN) branch for CausalWanSelfAttention.
# Two modes:
#   - "bidirectional": chunk-internal bidirectional + chunk-external causal
#     (matches DiT causality; pure matmul, no Triton kernels)
#   - "causal": token-level strict causal via existing Triton kernels from
#     extensions/GatedDeltaNet/

from typing import *
from torch import Tensor

import os
import sys
import math

import torch
from torch import nn
import torch.nn.functional as tF

from src.utils.distributed import get_sp_rank, get_sp_world_size, all_to_all


def l2_norm(x: Tensor):
    """
    Args:
        x: [b, l, d]
    """
    return x / (x.norm(dim=-1, keepdim=True) + 1e-5)


# ============================================================================
# Bidirectional mode: chunk-internal bidirectional, chunk-external causal
# ============================================================================

def _bidirectional_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gk: Tensor,
    chunk_size: int,
    initial_state: Tensor = None,
    output_final_state: bool = False,
):
    """
    Chunk-level bidirectional gated delta rule.

    Within each video chunk: all tokens see each other (bidirectional).
    Across chunks: causal (chunk i only sees chunks 0..i-1).

    Args:
        q: [B, H, L, D_k]
        k: [B, H, L, D_k]
        v: [B, H, L, D_v]
        beta: [B, H, L]  — update gate (sigmoid)
        gk: [B, H, L]    — log decay gate
        chunk_size: number of tokens per video chunk
        initial_state: [B, H, D_k, D_v] or None
        output_final_state: whether to return the final state

    Returns:
        o: [B, H, L, D_v]
        final_state: [B, H, D_k, D_v] or None
    """
    B, H, L, D_k = q.shape
    D_v = v.shape[-1]
    device, dtype = q.device, q.dtype

    # Ensure consistent dtypes (gates are computed in float32)
    beta = beta.to(dtype)
    gk = gk.float()  # keep gk in float for numerical stability in exp()

    # Pad to multiple of `chunk_size`
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad > 0:
        q = tF.pad(q, (0, 0, 0, pad))
        k = tF.pad(k, (0, 0, 0, pad))
        v = tF.pad(v, (0, 0, 0, pad))
        beta = tF.pad(beta, (0, pad))
        gk = tF.pad(gk, (0, pad))

    L_padded = q.shape[2]
    num_chunks = L_padded // chunk_size

    # Reshape into chunks: [B, H, num_chunks, chunk_size, ...]
    q_c = q.view(B, H, num_chunks, chunk_size, D_k)
    k_c = k.view(B, H, num_chunks, chunk_size, D_k)
    v_c = v.view(B, H, num_chunks, chunk_size, D_v)
    beta_c = beta.view(B, H, num_chunks, chunk_size)
    gk_c = gk.view(B, H, num_chunks, chunk_size)

    # State: [B, H, D_k, D_v]
    state = initial_state if initial_state is not None else \
        q.new_zeros(B, H, D_k, D_v)

    outputs = []
    for i in range(num_chunks):
        q_i = q_c[:, :, i]        # [B, H, C, D_k]
        k_i = k_c[:, :, i]        # [B, H, C, D_k]
        v_i = v_c[:, :, i]        # [B, H, C, D_v]
        beta_i = beta_c[:, :, i]  # [B, H, C]
        gk_i = gk_c[:, :, i]      # [B, H, C]

        # 1. Cross-chunk memory read (from previous state)
        o_mem = torch.einsum('bhcd,bhdv->bhcv', q_i, state)

        # 2. Intra-chunk bidirectional: all tokens see each other
        # KV_i = sum_j beta[j] * k[j] v[j]^T  -> [B, H, D_k, D_v]
        beta_kv = beta_i.unsqueeze(-1) * k_i  # [B, H, C, D_k]
        KV_i = torch.einsum('bhck,bhcv->bhkv', beta_kv, v_i)  # [B, H, D_k, D_v]
        o_local = torch.einsum('bhcd,bhdv->bhcv', q_i, KV_i)

        # 3. Output
        o_i = o_mem + o_local
        outputs.append(o_i)

        # 4. State update with gated delta rule
        # Chunk-level decay: product of per-token decays = exp(sum of log-decays)
        decay = gk_i.sum(dim=-1, keepdim=True).unsqueeze(-1).exp().to(dtype)  # [B, H, 1, 1]

        # Delta rule erasure: sum_j beta[j] * k[j] @ k[j]^T
        erase = torch.einsum('bhck,bhcd->bhkd', beta_kv, k_i)  # [B, H, D_k, D_k]

        # S_c = decay * (S_{c-1} - erase @ S_{c-1}) + KV_i
        state = decay * (state - torch.einsum('bhkd,bhdv->bhkv', erase, state)) + KV_i

    o = torch.stack(outputs, dim=2).view(B, H, L_padded, D_v)

    # Remove padding
    if pad > 0:
        o = o[:, :, :L]

    final_state = state if output_final_state else None
    return o, final_state


def _bidirectional_teacher_forcing(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gk: Tensor,
    chunk_size: int,
    clean_seq_len: int,
    initial_state: Tensor = None,
):
    """
    Teacher forcing variant of bidirectional mode.

    Sequence layout: [clean_tokens (clean_seq_len), noisy_tokens (clean_seq_len)].
    - Clean chunk i: reads state from clean 0..i-1, updates state.
    - Noisy chunk i: reads state from clean 0..i-1, does NOT update state.

    Args:
        q, k, v, beta, gk: [B, H, 2*clean_seq_len, ...]
        chunk_size: video chunk size in tokens
        clean_seq_len: number of clean tokens (= number of noisy tokens)
        initial_state: [B, H, D_k, D_v] or None

    Returns:
        o: [B, H, 2*clean_seq_len, D_v]
    """
    B, H = q.shape[:2]
    D_k = q.shape[-1]
    D_v = v.shape[-1]
    dtype = q.dtype

    # Ensure consistent dtypes
    beta = beta.to(dtype)
    gk = gk.float()

    # Split into clean and noisy
    q_clean, q_noisy = q[:, :, :clean_seq_len], q[:, :, clean_seq_len:]
    k_clean = k[:, :, :clean_seq_len]
    v_clean = v[:, :, :clean_seq_len]
    beta_clean = beta[:, :, :clean_seq_len]
    gk_clean = gk[:, :, :clean_seq_len]

    # Pad clean tokens to multiple of `chunk_size`
    L_clean = clean_seq_len
    pad = (chunk_size - L_clean % chunk_size) % chunk_size
    if pad > 0:
        q_clean = tF.pad(q_clean, (0, 0, 0, pad))
        k_clean = tF.pad(k_clean, (0, 0, 0, pad))
        v_clean = tF.pad(v_clean, (0, 0, 0, pad))
        beta_clean = tF.pad(beta_clean, (0, pad))
        gk_clean = tF.pad(gk_clean, (0, pad))
        q_noisy = tF.pad(q_noisy, (0, 0, 0, pad))

    L_padded = q_clean.shape[2]
    num_chunks = L_padded // chunk_size

    # Reshape into chunks
    q_clean_c = q_clean.view(B, H, num_chunks, chunk_size, D_k)
    k_clean_c = k_clean.view(B, H, num_chunks, chunk_size, D_k)
    v_clean_c = v_clean.view(B, H, num_chunks, chunk_size, D_v)
    beta_clean_c = beta_clean.view(B, H, num_chunks, chunk_size)
    gk_clean_c = gk_clean.view(B, H, num_chunks, chunk_size)
    q_noisy_c = q_noisy.view(B, H, num_chunks, chunk_size, D_k)

    state = initial_state if initial_state is not None else \
        q.new_zeros(B, H, D_k, D_v)

    out_clean_list = []
    out_noisy_list = []

    for i in range(num_chunks):
        q_ci = q_clean_c[:, :, i]
        k_ci = k_clean_c[:, :, i]
        v_ci = v_clean_c[:, :, i]
        beta_ci = beta_clean_c[:, :, i]
        gk_ci = gk_clean_c[:, :, i]
        q_ni = q_noisy_c[:, :, i]

        # Noisy chunk i: read-only from state (clean 0..i-1), no update
        o_noisy_mem = torch.einsum('bhcd,bhdv->bhcv', q_ni, state)
        out_noisy_list.append(o_noisy_mem)

        # Clean chunk i: full processing + state update
        # Cross-chunk read
        o_clean_mem = torch.einsum('bhcd,bhdv->bhcv', q_ci, state)
        # Intra-chunk bidirectional
        beta_kv = beta_ci.unsqueeze(-1) * k_ci
        KV_i = torch.einsum('bhck,bhcv->bhkv', beta_kv, v_ci)
        o_clean_local = torch.einsum('bhcd,bhdv->bhcv', q_ci, KV_i)
        out_clean_list.append(o_clean_mem + o_clean_local)

        # State update
        decay = gk_ci.sum(dim=-1, keepdim=True).unsqueeze(-1).exp().to(dtype)
        erase = torch.einsum('bhck,bhcd->bhkd', beta_kv, k_ci)
        state = decay * (state - torch.einsum('bhkd,bhdv->bhkv', erase, state)) + KV_i

    # Stack and remove padding
    o_clean = torch.stack(out_clean_list, dim=2).view(B, H, L_padded, D_v)
    o_noisy = torch.stack(out_noisy_list, dim=2).view(B, H, L_padded, D_v)
    if pad > 0:
        o_clean = o_clean[:, :, :L_clean]
        o_noisy = o_noisy[:, :, :L_clean]

    # Reconstruct full output: [clean, noisy]
    o = torch.cat([o_clean, o_noisy], dim=2)
    return o


# ============================================================================
# Causal mode: load Triton kernels from extensions/GatedDeltaNet
# ============================================================================

def _get_causal_kernel():
    """Lazy-import the Triton kernel from extensions/GatedDeltaNet."""
    ext_path = os.path.join(
        os.path.dirname(__file__), "../../../../extensions/GatedDeltaNet/lit_gpt")
    if ext_path not in sys.path:
        sys.path.insert(0, ext_path)
    from gated_delta_rule_ops import chunk_gated_delta_rule
    return chunk_gated_delta_rule


def _causal_teacher_forcing(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gk: Tensor,
    chunk_size: int,
    clean_seq_len: int,
    initial_state: Tensor = None,
):
    """
    Teacher forcing for token-level causal mode.

    Same semantics as bidirectional teacher forcing: noisy chunk i sees
    state from clean 0..i-1 only, does not update state.

    In causal mode, clean chunks go through full `chunk_gated_delta_rule`
    (token-level causal within chunk + state update), noisy chunks do
    read-only `q @ state`.
    """
    chunk_gated_delta_rule = _get_causal_kernel()

    B, H = q.shape[:2]
    D_k = q.shape[-1]
    D_v = v.shape[-1]
    dtype = q.dtype

    # Cast beta to match q/v dtype for einsum compatibility
    beta = beta.to(dtype)

    q_clean, q_noisy = q[:, :, :clean_seq_len], q[:, :, clean_seq_len:]
    k_clean = k[:, :, :clean_seq_len]
    v_clean = v[:, :, :clean_seq_len]
    beta_clean = beta[:, :, :clean_seq_len]
    gk_clean = gk[:, :, :clean_seq_len]

    # Pad to multiple of `chunk_size`
    L_clean = clean_seq_len
    pad = (chunk_size - L_clean % chunk_size) % chunk_size
    if pad > 0:
        q_clean = tF.pad(q_clean, (0, 0, 0, pad))
        k_clean = tF.pad(k_clean, (0, 0, 0, pad))
        v_clean = tF.pad(v_clean, (0, 0, 0, pad))
        beta_clean = tF.pad(beta_clean, (0, pad))
        gk_clean = tF.pad(gk_clean, (0, pad))
        q_noisy = tF.pad(q_noisy, (0, 0, 0, pad))

    L_padded = q_clean.shape[2]
    num_chunks = L_padded // chunk_size

    # Reshape into video chunks
    q_clean_c = q_clean.view(B, H, num_chunks, chunk_size, D_k)
    k_clean_c = k_clean.view(B, H, num_chunks, chunk_size, D_k)
    v_clean_c = v_clean.view(B, H, num_chunks, chunk_size, D_v)
    beta_clean_c = beta_clean.view(B, H, num_chunks, chunk_size)
    gk_clean_c = gk_clean.view(B, H, num_chunks, chunk_size)
    q_noisy_c = q_noisy.view(B, H, num_chunks, chunk_size, D_k)

    state = initial_state if initial_state is not None else \
        q.new_zeros(B, H, D_k, D_v)

    out_clean_list = []
    out_noisy_list = []

    for i in range(num_chunks):
        q_ni = q_noisy_c[:, :, i]

        # Noisy chunk i: read-only
        o_noisy = torch.einsum('bhcd,bhdv->bhcv', q_ni, state)
        out_noisy_list.append(o_noisy)

        # Clean chunk i: full causal processing
        # `chunk_gated_delta_rule` expects [B, H, L, D] with L being the chunk
        o_clean, state = chunk_gated_delta_rule(
            q_clean_c[:, :, i], k_clean_c[:, :, i], v_clean_c[:, :, i],
            beta_clean_c[:, :, i], gk_clean_c[:, :, i],
            initial_state=state, output_final_state=True)
        out_clean_list.append(o_clean)

    o_clean = torch.stack(out_clean_list, dim=2).view(B, H, L_padded, D_v)
    o_noisy = torch.stack(out_noisy_list, dim=2).view(B, H, L_padded, D_v)
    if pad > 0:
        o_clean = o_clean[:, :, :L_clean]
        o_noisy = o_noisy[:, :, :L_clean]

    return torch.cat([o_clean, o_noisy], dim=2)


# ============================================================================
# GDNBranch — the main module to be integrated into CausalWanSelfAttention
# ============================================================================

class GDNBranch(nn.Module):
    """
    GatedDeltaNet branch that runs in parallel with attention inside
    CausalWanSelfAttention. Reuses Q/K/V projections from attention; output
    is added to attention output before `o_proj`.

    The `gdn_scale_proj` is zero-initialized so the branch produces zero
    output at initialization, preserving the pretrained model's behavior.

    Two causal modes:
      - "bidirectional": chunk-internal bidirectional, chunk-external causal
        (matches DiT causality structure)
      - "causal": token-level strict causal via Triton kernels
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_gdn_heads: int = 4,
        head_qk_dim: Optional[int] = None,
        head_v_dim: Optional[int] = None,
        causal_mode: str = "bidirectional",
        chunk_size: int = 4680,  # default: `frame_seqlen * chunk_size`
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_gdn_heads = num_gdn_heads
        self.head_qk_dim = head_qk_dim if head_qk_dim is not None else self.head_dim
        self.head_v_dim = head_v_dim if head_v_dim is not None else self.head_qk_dim
        self.causal_mode = causal_mode
        self.chunk_size = chunk_size

        assert causal_mode in ("bidirectional", "causal"), \
            f"Unknown `causal_mode`: {causal_mode}"

        # Dimensions for slicing from attention Q/K/V
        self.gdn_qk_dim = num_gdn_heads * self.head_qk_dim
        self.gdn_v_dim = num_gdn_heads * self.head_v_dim

        # ---------- GDN-specific projections (lightweight) ----------
        # Decay gate (Mamba-style): gk = -A.exp() * softplus(gk_proj + dt_bias)
        self.gk_proj = nn.Linear(dim, num_gdn_heads, bias=False)
        A = torch.empty(num_gdn_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(num_gdn_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # Update gate beta: sigmoid(b_proj)
        self.b_proj = nn.Linear(dim, num_gdn_heads, bias=True)

        # Output gate
        self.g_proj = nn.Linear(dim, self.gdn_v_dim, bias=False)

        # Output normalization (per-head RMSNorm)
        self.gdn_norm = nn.RMSNorm(self.head_v_dim, eps=1e-5, elementwise_affine=True)

        # Learnable GDN output scale (zero-initialized — produces zero output at init)
        self.gdn_scale_proj = nn.Linear(dim, num_gdn_heads)
        nn.init.zeros_(self.gdn_scale_proj.weight)
        nn.init.zeros_(self.gdn_scale_proj.bias)

        # Output projection (zero-initialized)
        self.o_proj = nn.Linear(self.gdn_v_dim, dim, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def _prepare_qkv(self, q: Tensor, k: Tensor, v: Tensor):
        """
        Slice and reshape Q/K/V for GDN heads.

        Args:
            q, k, v: [B, L, num_heads, head_dim] from CausalWanSelfAttention

        Returns:
            gdn_q, gdn_k: [B, num_gdn_heads, L, head_qk_dim]
            gdn_v: [B, num_gdn_heads, L, head_v_dim]
        """
        B, L = q.shape[0], q.shape[1]

        # Flatten heads, slice first `gdn_qk_dim` / `gdn_v_dim` dims
        q_flat = q.flatten(2)[:, :, :self.gdn_qk_dim]
        k_flat = k.flatten(2)[:, :, :self.gdn_qk_dim]
        v_flat = v.flatten(2)[:, :, :self.gdn_v_dim]

        # Reshape: [B, L, gdn_dim] -> [B, num_gdn_heads, L, head_dim]
        gdn_q = q_flat.view(B, L, self.num_gdn_heads, self.head_qk_dim).permute(0, 2, 1, 3)
        gdn_k = k_flat.view(B, L, self.num_gdn_heads, self.head_qk_dim).permute(0, 2, 1, 3)
        gdn_v = v_flat.view(B, L, self.num_gdn_heads, self.head_v_dim).permute(0, 2, 1, 3)

        # L2 normalize Q and K
        gdn_q = l2_norm(gdn_q)
        gdn_k = l2_norm(gdn_k)

        return gdn_q, gdn_k, gdn_v

    def _compute_gates(self, hidden_states: Tensor):
        """
        Compute decay gate (gk) and update gate (beta).

        Args:
            hidden_states: [B, L, dim]

        Returns:
            gk: [B, num_gdn_heads, L]   — log-space decay
            beta: [B, num_gdn_heads, L]  — update gate in (0, 1)
        """
        gk = self.gk_proj(hidden_states).float()  # [B, L, num_gdn_heads]
        gk = -self.A_log.float().exp() * tF.softplus(gk + self.dt_bias)
        gk = gk.transpose(1, 2)  # [B, num_gdn_heads, L]

        beta = self.b_proj(hidden_states).float().sigmoid()
        beta = beta.transpose(1, 2)  # [B, num_gdn_heads, L]

        return gk, beta

    def _apply_output_gate_and_norm(self, o: Tensor, hidden_states: Tensor):
        """
        Apply RMSNorm, output gate, scale, and output projection.

        Args:
            o: [B, num_gdn_heads, L, head_v_dim]
            hidden_states: [B, L, dim]

        Returns:
            output: [B, L, dim]
        """
        B, _, L, _ = o.shape

        # Per-head RMSNorm: [B, num_gdn_heads, L, head_v_dim] -> [B, L, num_gdn_heads, head_v_dim]
        o = o.permute(0, 2, 1, 3)  # [B, L, H_gdn, D_v]
        o = self.gdn_norm(o)

        # Output gate: swish(g_proj(x))
        g = tF.silu(self.g_proj(hidden_states))  # [B, L, gdn_v_dim]
        g = g.view(B, L, self.num_gdn_heads, self.head_v_dim)
        o = o * g

        # Zero-init scale
        scale = tF.silu(self.gdn_scale_proj(hidden_states))  # [B, L, num_gdn_heads]
        o = o * scale.unsqueeze(-1)

        # Flatten heads and project
        o = o.reshape(B, L, self.gdn_v_dim)
        o = self.o_proj(o)

        return o

    def reset_parameters(self):
        # Required by FSDP to materialize meta-device parameters
        pass

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        hidden_states: Tensor,
        #
        gdn_state: Optional[dict] = None,
        #
        teacher_forcing_clean_len: Optional[int] = None,
    ):
        """
        Forward pass of the GDN branch.

        Args:
            q: [B, L, num_heads, head_dim]; shared with attention (pre-RoPE)
               With SP, L is the local shard length (L_full // sp_size)
            k: [B, L, num_heads, head_dim]
            v: [B, L, num_heads, head_dim]
            hidden_states: [B, L, dim]; input to self-attention (for gate projections)

            gdn_state: dict or None; for inference, carries recurrent state across chunks.
                       Mutated in-place (like KV-cache).

            teacher_forcing_clean_len: int or None; total number of clean tokens
                when the sequence layout is [clean, noisy]. If provided, uses
                teacher-forcing logic that only updates state from clean tokens.

        Returns:
            gdn_output: [B, L, dim]
        """
        B, L_local = q.shape[0], q.shape[1]
        sp_size = get_sp_world_size()
        use_sp = sp_size > 1

        # ==================================================================
        # Step 1: Prepare Q/K/V and gates for GDN heads
        # With SP: gather sequence, scatter heads (Ulysses-style)
        # ==================================================================
        if use_sp:
            assert self.num_gdn_heads % sp_size == 0, \
                f"`num_gdn_heads` ({self.num_gdn_heads}) must be divisible by `sp_size` ({sp_size})"
            local_gdn_heads = self.num_gdn_heads // sp_size
            sp_rank = get_sp_rank()

            # Extract GDN heads from Q/K/V
            q_gdn = q.flatten(2)[:, :, :self.gdn_qk_dim].view(
                B, L_local, self.num_gdn_heads, self.head_qk_dim)
            k_gdn = k.flatten(2)[:, :, :self.gdn_qk_dim].view(
                B, L_local, self.num_gdn_heads, self.head_qk_dim)
            v_gdn = v.flatten(2)[:, :, :self.gdn_v_dim].view(
                B, L_local, self.num_gdn_heads, self.head_v_dim)

            # all-to-all: [B, L_local, num_gdn_heads, d] -> [B, L_full, local_gdn_heads, d]
            q_gdn = all_to_all(q_gdn, scatter_dim=2, gather_dim=1)
            k_gdn = all_to_all(k_gdn, scatter_dim=2, gather_dim=1)
            v_gdn = all_to_all(v_gdn, scatter_dim=2, gather_dim=1)
            L = q_gdn.shape[1]

            # [B, L, local_heads, d] -> [B, local_heads, L, d]
            gdn_q = l2_norm(q_gdn.permute(0, 2, 1, 3))
            gdn_k = l2_norm(k_gdn.permute(0, 2, 1, 3))
            gdn_v = v_gdn.permute(0, 2, 1, 3)

            # Gates: compute on local shard, then all-to-all
            gk_full = self.gk_proj(hidden_states).float()  # [B, L_local, num_gdn_heads]
            gk_full = -self.A_log.float().exp() * tF.softplus(gk_full + self.dt_bias)
            gk_full = gk_full.view(B, L_local, self.num_gdn_heads, 1)
            gk_full = all_to_all(gk_full, scatter_dim=2, gather_dim=1)  # [B, L, local_heads, 1]
            gk = gk_full.squeeze(-1).permute(0, 2, 1)  # [B, local_heads, L]

            beta_full = self.b_proj(hidden_states).float().sigmoid()  # [B, L_local, num_gdn_heads]
            beta_full = beta_full.view(B, L_local, self.num_gdn_heads, 1)
            beta_full = all_to_all(beta_full, scatter_dim=2, gather_dim=1)  # [B, L, local_heads, 1]
            beta = beta_full.squeeze(-1).permute(0, 2, 1)  # [B, local_heads, L]

            active_heads = local_gdn_heads
        else:
            gdn_q, gdn_k, gdn_v = self._prepare_qkv(q, k, v)
            gk, beta = self._compute_gates(hidden_states)
            L = L_local
            active_heads = self.num_gdn_heads

        # ==================================================================
        # Step 2: Run GDN kernel
        # ==================================================================
        if gdn_state is None:
            # Training
            initial_state = None

            if teacher_forcing_clean_len is not None:
                # Teacher forcing mode
                if self.causal_mode == "bidirectional":
                    o = _bidirectional_teacher_forcing(
                        gdn_q, gdn_k, gdn_v, beta, gk,
                        chunk_size=self.chunk_size,
                        clean_seq_len=teacher_forcing_clean_len,
                        initial_state=initial_state,
                    )
                else:
                    o = _causal_teacher_forcing(
                        gdn_q, gdn_k, gdn_v, beta, gk,
                        chunk_size=self.chunk_size,
                        clean_seq_len=teacher_forcing_clean_len,
                        initial_state=initial_state,
                    )
            else:
                # Standard forward
                if self.causal_mode == "bidirectional":
                    o, _ = _bidirectional_forward(
                        gdn_q, gdn_k, gdn_v, beta, gk,
                        chunk_size=self.chunk_size,
                        initial_state=initial_state,
                        output_final_state=False,
                    )
                else:
                    chunk_gated_delta_rule = _get_causal_kernel()
                    o, _ = chunk_gated_delta_rule(
                        gdn_q, gdn_k, gdn_v, beta, gk,
                        initial_state=initial_state,
                        output_final_state=False,
                    )
        else:
            # Inference: use carried state
            recurrent_state = gdn_state["recurrent_state"]

            if self.causal_mode == "bidirectional":
                o, new_state = _bidirectional_forward(
                    gdn_q, gdn_k, gdn_v, beta, gk,
                    chunk_size=self.chunk_size,
                    initial_state=recurrent_state,
                    output_final_state=True,
                )
            else:
                chunk_gated_delta_rule = _get_causal_kernel()
                o, new_state = chunk_gated_delta_rule(
                    gdn_q, gdn_k, gdn_v, beta, gk,
                    initial_state=recurrent_state,
                    output_final_state=True,
                )

            # Mutate state in-place
            gdn_state["recurrent_state"] = new_state

        # ==================================================================
        # Step 3: Output gate + norm + scale, then reverse SP if needed
        # ==================================================================
        if use_sp:
            # o: [B, local_heads, L, head_v_dim]
            # Transpose to [B, L, local_heads, head_v_dim] for norm
            o = o.permute(0, 2, 1, 3)
            o = self.gdn_norm(o)

            # Output gate: compute on local shard, then all-to-all
            g = tF.silu(self.g_proj(hidden_states))  # [B, L_local, gdn_v_dim]
            g = g.view(B, L_local, self.num_gdn_heads, self.head_v_dim)
            g = all_to_all(g, scatter_dim=2, gather_dim=1)  # [B, L, local_heads, head_v_dim]
            o = o * g

            # Scale: compute on local shard, then all-to-all
            scale = tF.silu(self.gdn_scale_proj(hidden_states))  # [B, L_local, num_gdn_heads]
            scale = scale.view(B, L_local, self.num_gdn_heads, 1)
            scale = all_to_all(scale, scatter_dim=2, gather_dim=1)  # [B, L, local_heads, 1]
            o = o * scale

            # Reverse all-to-all: [B, L, local_heads, d] -> [B, L_local, num_gdn_heads, d]
            o = all_to_all(o, scatter_dim=1, gather_dim=2)  # [B, L_local, num_gdn_heads, head_v_dim]
            o = o.reshape(B, L_local, self.gdn_v_dim)
            o = self.o_proj(o)
        else:
            # o: [B, num_gdn_heads, L, head_v_dim]
            o = self._apply_output_gate_and_norm(o, hidden_states)

        return o

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Initialize GDN state for inference (called once before autoregressive generation).
        With SP, only initializes state for this rank's local heads.
        """
        sp_size = get_sp_world_size()

        if sp_size > 1:
            assert self.num_gdn_heads % sp_size == 0
            local_heads = self.num_gdn_heads // sp_size
        else:
            local_heads = self.num_gdn_heads

        recurrent_state = torch.zeros(
            batch_size, local_heads, self.head_qk_dim, self.head_v_dim,
            device=device, dtype=dtype)

        return {"recurrent_state": recurrent_state}
