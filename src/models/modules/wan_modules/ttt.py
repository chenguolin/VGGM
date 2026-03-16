# TTT (Test-Time Training) branch for CausalWanSelfAttention.
# Adapted from Spatial-TTT (https://github.com/KMnP/Spatial-TTT) with SP support.

from typing import *

import os
import math

import torch
from torch import nn
import torch.nn.functional as tF

from src.utils.distributed import get_sp_rank, get_sp_world_size, all_to_all

# Set TTT_NO_COMPILE=1 to disable @torch.compile on TTT kernels.
# Useful for debugging gradient flow (torch.compile can break autograd
# with in-place ops like slice assignment and tF.silu(inplace=True)).
_ttt_compile = (lambda fn: fn) if os.environ.get("TTT_NO_COMPILE") else torch.compile()


def inv_softplus(x: float) -> float:
    return math.log(math.exp(x) - 1.0)


# ============================================================================
# TTT operations (from Spatial-TTT ttt_operation.py)
# ============================================================================

@_ttt_compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@_ttt_compile
def l2_norm(x: torch.Tensor):
    """
    Args:
        x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
    return ret.type(x_type)


@_ttt_compile
def zeropower_via_newtonschulz5(G):
    """
    Newton-Schulz iteration for orthogonalization (Muon optimizer).

    Args:
        G: [b, d, d']
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@_ttt_compile
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def prenorm_block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
):
    """
    Block causal LaCT with SwiGLU fast weight function (prenorm variant).
    Apply then Update => Shifted Block Causal LaCT.

    f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))

    Args:
        w0: [B*num_fw_heads, d_h, d_in] fast weight (fp32)
        w1: [B*num_fw_heads, d_out, d_h] fast weight (fp32)
        w2: [B*num_fw_heads, d_h, d_in] fast weight (fp32)
        q:  [B*num_fw_heads, L, d_in]
        k:  [B*num_fw_heads, L, d_in]
        v:  [B*num_fw_heads, L, d_out]
        lr0, lr1, lr2: [B*num_fw_heads, L, d/1] per-token learning rates (fp32)
        chunk_size: TTT update granularity
        use_muon: enable Newton-Schulz orthogonalization
        momentum: [B*num_fw_heads, L, 1] optional momentum
    Returns:
        output: [B*num_fw_heads, L, d_out]
    """
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, d, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        qi = q[:, :, s_index:e_index]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]

        # Apply: use current fast weights to compute output for current chunk
        h = torch.bmm(w2, qi)
        gate = tF.silu(torch.bmm(w0, qi), inplace=True)
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # Forward pass with key (for gradient computation)
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        hidden = tF.silu(gate_before_act, inplace=False) * hidden_before_mul

        # Backward pass: compute gradients w.r.t. fast weights
        dhidden = torch.bmm(w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * tF.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        # Update fast weights (prenorm: accumulate on unnormalized, apply norm for next iteration)
        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2

        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # Last chunk: apply only, no update
    s_index = e_index
    e_index = seq_len
    qi = q[:, :, s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = tF.silu(torch.bmm(w0, qi), inplace=True)
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


@_ttt_compile
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
):
    """
    Block causal LaCT with SwiGLU fast weight function (postnorm variant).
    Same interface as `prenorm_block_causal_lact_swiglu`.
    """
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        qi = q[:, :, s_index:e_index]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        lr0i = lr0[:, s_index:e_index, :]

        h = torch.bmm(w2, qi)
        gate = tF.silu(torch.bmm(w0, qi), inplace=True)
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
        hidden = tF.silu(gate_before_act, inplace=False) * hidden_before_mul
        dhidden = torch.bmm(w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * tF.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)
            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2

        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    s_index = e_index
    e_index = seq_len
    qi = q[:, :, s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = tF.silu(torch.bmm(w0, qi), inplace=True)
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def ttt_apply_only(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
):
    """
    Apply fast weights to query without updating (for inference single-chunk).

    Args:
        w0: [B*num_fw_heads, d_h, d_in]
        w1: [B*num_fw_heads, d_out, d_h]
        w2: [B*num_fw_heads, d_h, d_in]
        q:  [B*num_fw_heads, L, d_in]
    Returns:
        output: [B*num_fw_heads, L, d_out]
    """
    q_t = q.transpose(1, 2)  # [b, d_in, l]
    h = torch.bmm(w2, q_t)
    gate = tF.silu(torch.bmm(w0, q_t), inplace=True)
    output = torch.bmm(w1, gate * h)
    return output.transpose(1, 2)


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def ttt_update_state(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
    dw0_momentum: torch.Tensor = None,
    dw1_momentum: torch.Tensor = None,
    dw2_momentum: torch.Tensor = None,
    prenorm: bool = True,
    w0_main: torch.Tensor = None,
    w1_main: torch.Tensor = None,
    w2_main: torch.Tensor = None,
    w0_norm: torch.Tensor = None,
    w1_norm: torch.Tensor = None,
    w2_norm: torch.Tensor = None,
):
    """
    Update TTT fast weight state given a new chunk's K/V (for inference).
    Processes the entire chunk as a single update step.

    Returns updated (w0, w1, w2) and optionally updated momentum tensors.
    """
    # Forward pass with key
    gate_before_act = torch.bmm(w0, k.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, k.transpose(1, 2))
    hidden = tF.silu(gate_before_act, inplace=False) * hidden_before_mul

    # Backward pass
    v_t = v.transpose(1, 2)
    dhidden = torch.bmm(w1.transpose(1, 2), v_t)
    dhidden_before_mul = dhidden * tF.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(v_t, (hidden.transpose(1, 2) * lr1).type_as(v_t))
    dw0 = torch.bmm(dgate_before_act, (k * lr0).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (k * lr2).type_as(dhidden_before_mul))

    if momentum is not None and dw0_momentum is not None:
        m = momentum.mean(dim=1, keepdim=True)
        dw0 = dw0 + dw0_momentum * m
        dw1 = dw1 + dw1_momentum * m
        dw2 = dw2 + dw2_momentum * m

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw2 = zeropower_via_newtonschulz5(dw2)

    if prenorm:
        w0_main = w0_main + dw0
        w1_main = w1_main + dw1
        w2_main = w2_main + dw2
        w0_out = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1_out = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2_out = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
    else:
        w0_out = w0 + dw0
        w1_out = w1 + dw1
        w2_out = w2 + dw2
        norm0 = w0.norm(dim=2, keepdim=True)
        norm1 = w1.norm(dim=2, keepdim=True)
        norm2 = w2.norm(dim=2, keepdim=True)
        w0_out = w0_out / (w0_out.norm(dim=2, keepdim=True) + 1e-5) * norm0
        w1_out = w1_out / (w1_out.norm(dim=2, keepdim=True) + 1e-5) * norm1
        w2_out = w2_out / (w2_out.norm(dim=2, keepdim=True) + 1e-5) * norm2
        w0_main = w0_out
        w1_main = w1_out
        w2_main = w2_out

    return w0_out, w1_out, w2_out, dw0, dw1, dw2, w0_main, w1_main, w2_main


# ============================================================================
# Low-rank fast weight parameterization
# ============================================================================

class LowRankFastWeight(nn.Module):
    def __init__(self, num_heads, out_features, in_features, rank,
                 init_gain=0.5, add_identity=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity

        self.w_left = nn.Parameter(torch.empty(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.empty(num_heads, rank, in_features))
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(rank) * init_gain)
        nn.init.normal_(self.w_right, std=1.0 / math.sqrt(in_features) * init_gain)

    def forward(self):
        W = self.w_left @ self.w_right
        if self.add_identity:
            W = W + 0.5 * torch.eye(
                self.out_features, self.in_features,
                device=W.device, dtype=W.dtype,
            ).unsqueeze(0)
        return W


# ============================================================================
# TTTBranch — the main module to be integrated into CausalWanSelfAttention
# ============================================================================

class TTTBranch(nn.Module):
    """
    TTT branch that runs in parallel with attention inside CausalWanSelfAttention.
    Shares Q/K/V projections with attention; output is added to attention output
    before `o_proj`.

    The `ttt_scale_proj` is zero-initialized so the branch produces zero output
    at initialization, preserving the pretrained model's behavior.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_fw_heads: int = 4,
        fw_head_dim: Optional[int] = None,
        ttt_chunk_size: int = 4680,  # default: frame_seqlen * chunk_size
        w0_w2_low_rank: int = 32,
        use_muon: bool = True,
        use_momentum: bool = True,
        prenorm: bool = True,
        fp32_states: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_fw_heads = num_fw_heads
        self.fw_head_dim = fw_head_dim if fw_head_dim is not None else self.head_dim
        self.ttt_chunk_size = ttt_chunk_size
        self.w0_w2_low_rank = w0_w2_low_rank
        self.use_muon = use_muon
        self.use_momentum = use_momentum
        self.prenorm = prenorm
        self.fp32_states = fp32_states

        # Dimension of TTT Q/K/V slice from the full Q/K/V
        self.ttt_dim = num_fw_heads * self.fw_head_dim

        d_in = self.fw_head_dim
        d_out = self.fw_head_dim
        d_h = d_in  # `inter_multi` = 1.0

        # Fast weights: f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))
        if w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(num_fw_heads, d_h, d_in, w0_w2_low_rank)
            self.w2 = LowRankFastWeight(num_fw_heads, d_h, d_in, w0_w2_low_rank)
        else:
            self.w0 = nn.Parameter(
                torch.randn(num_fw_heads, d_h, d_in) / math.sqrt(d_in))
            self.w2 = nn.Parameter(
                torch.randn(num_fw_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(
            torch.randn(num_fw_heads, d_out, d_h) / math.sqrt(d_h))

        # Per-token learning rate projection
        lr_dim = 3 * num_fw_heads  # `lr_dim_per_head` = 1
        self.lr_proj = nn.Linear(dim, lr_dim)
        self.base_lr_inv = inv_softplus(0.001)

        # Learnable TTT output scale (zero-initialized gate)
        self.ttt_scale_proj = nn.Linear(dim, num_fw_heads)
        nn.init.zeros_(self.ttt_scale_proj.weight)
        nn.init.zeros_(self.ttt_scale_proj.bias)

        # Output normalization
        self.ttt_norm = nn.RMSNorm(self.fw_head_dim, eps=1e-5, elementwise_affine=True)

        # Momentum
        if use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(dim, num_fw_heads),
                nn.Sigmoid(),
            )

    def _get_base_weights(self):
        """Get base fast weights, handling low-rank parameterization."""
        if self.w0_w2_low_rank > 0:
            w0 = self.w0()
            w2 = self.w2()
        else:
            w0 = self.w0
            w2 = self.w2
        w1 = self.w1
        return w0, w1, w2

    def _prepare_qkv(self, q, k, v):
        """
        Slice and reshape Q/K/V for TTT heads.

        Args:
            q, k, v: [B, L, num_heads, head_dim] from CausalWanSelfAttention
        Returns:
            fast_q, fast_k, fast_v: [B*num_fw_heads, L, fw_head_dim]
        """
        B, L = q.shape[0], q.shape[1]

        # Flatten heads then take first `ttt_dim` dimensions
        q_flat = q.flatten(2)[:, :, :self.ttt_dim]  # [B, L, ttt_dim]
        k_flat = k.flatten(2)[:, :, :self.ttt_dim]
        v_flat = v.flatten(2)[:, :, :self.ttt_dim]

        # Reshape to [B*num_fw_heads, L, fw_head_dim]
        fast_q = q_flat.view(B, L, self.num_fw_heads, self.fw_head_dim) \
            .permute(0, 2, 1, 3).reshape(B * self.num_fw_heads, L, self.fw_head_dim)
        fast_k = k_flat.view(B, L, self.num_fw_heads, self.fw_head_dim) \
            .permute(0, 2, 1, 3).reshape(B * self.num_fw_heads, L, self.fw_head_dim)
        fast_v = v_flat.view(B, L, self.num_fw_heads, self.fw_head_dim) \
            .permute(0, 2, 1, 3).reshape(B * self.num_fw_heads, L, self.fw_head_dim)

        # L2 normalize Q and K
        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        return fast_q, fast_k, fast_v

    def _compute_lr_and_momentum(self, hidden_states):
        """
        Compute per-token learning rates and optional momentum.

        Args:
            hidden_states: [B, L, dim]
        Returns:
            lr0, lr1, lr2: [B*num_fw_heads, L, 1]
            momentum: [B*num_fw_heads, L, 1] or None
        """
        B, L = hidden_states.shape[:2]

        lr = self.lr_proj(hidden_states)  # [B, L, 3*num_fw_heads]
        lr = tF.softplus(lr.float() + self.base_lr_inv)  # [B, L, 3*num_fw_heads]
        lr = lr.view(B, L, self.num_fw_heads, 3) \
            .permute(0, 2, 1, 3).reshape(B * self.num_fw_heads, L, 3)
        lr0, lr1, lr2 = lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()  # [B, L, num_fw_heads]
            momentum = momentum.view(B, L, self.num_fw_heads, 1) \
                .permute(0, 2, 1, 3).reshape(B * self.num_fw_heads, L, 1)
        else:
            momentum = None

        return lr0, lr1, lr2, momentum

    def _apply_scale_and_norm(self, fw_x, hidden_states):
        """
        Apply RMSNorm and learnable scale to TTT output.

        Args:
            fw_x: [B*num_fw_heads, L, fw_head_dim]
            hidden_states: [B, L, dim]
        Returns:
            ttt_output: [B, L, ttt_dim]
        """
        B, L = hidden_states.shape[:2]

        # RMSNorm per head
        ttt_x_normed = self.ttt_norm(fw_x)  # [B*num_fw_heads, L, fw_head_dim]

        # Learnable scale (zero-initialized)
        ttt_scale = tF.silu(self.ttt_scale_proj(hidden_states))  # [B, L, num_fw_heads]
        ttt_scale = ttt_scale.view(B, L, self.num_fw_heads, 1) \
            .permute(0, 2, 1, 3).reshape(B * self.num_fw_heads, L, 1)
        ttt_x_normed = ttt_x_normed * ttt_scale

        # Reshape back: [B*num_fw_heads, L, fw_head_dim] -> [B, L, ttt_dim]
        ttt_output = ttt_x_normed.view(B, self.num_fw_heads, L, self.fw_head_dim) \
            .permute(0, 2, 1, 3).reshape(B, L, self.ttt_dim)

        return ttt_output

    def forward(self, q, k, v, hidden_states, ttt_state=None):
        """
        Forward pass of the TTT branch.

        Args:
            q: [B, L, num_heads, head_dim] — shared with attention (pre-RoPE).
               With SP, L is the local shard length (L_full // sp_size).
            k: [B, L, num_heads, head_dim]
            v: [B, L, num_heads, head_dim]
            hidden_states: [B, L, dim] — input to self-attention (for lr/scale projections)
            ttt_state: dict or None — for inference, carries fast weights across chunks.
                       Mutated **in-place** (like KV-cache).

        Returns:
            ttt_output: [B, L, dim] — zero-padded to full `dim`
        """
        B, L_local = q.shape[0], q.shape[1]
        sp_size = get_sp_world_size()
        use_sp = sp_size > 1

        # ==================================================================
        # Step 1: Prepare Q/K/V, lr, momentum for TTT heads
        #
        # With SP (Ulysses-style): gather sequence, scatter heads so each
        # rank sees the full sequence but only `local_fw_heads` TTT heads.
        # Without SP: use the existing helper methods directly.
        # ==================================================================
        if use_sp:
            assert self.num_fw_heads % sp_size == 0, \
                f"`num_fw_heads` ({self.num_fw_heads}) must be divisible by `sp_size` ({sp_size})"
            local_fw_heads = self.num_fw_heads // sp_size
            sp_rank = get_sp_rank()

            # Extract TTT heads from Q/K/V
            # [B, L_local, num_heads, head_dim] -> [B, L_local, num_fw_heads, fw_head_dim]
            q_ttt = q.flatten(2)[:, :, :self.ttt_dim].view(
                B, L_local, self.num_fw_heads, self.fw_head_dim)
            k_ttt = k.flatten(2)[:, :, :self.ttt_dim].view(
                B, L_local, self.num_fw_heads, self.fw_head_dim)
            v_ttt = v.flatten(2)[:, :, :self.ttt_dim].view(
                B, L_local, self.num_fw_heads, self.fw_head_dim)

            # all-to-all: [B, L_local, num_fw_heads, d] -> [B, L_full, local_fw_heads, d]
            q_ttt = all_to_all(q_ttt, scatter_dim=2, gather_dim=1)
            k_ttt = all_to_all(k_ttt, scatter_dim=2, gather_dim=1)
            v_ttt = all_to_all(v_ttt, scatter_dim=2, gather_dim=1)
            L = q_ttt.shape[1]

            # [B, L, local_fw_heads, d] -> [B*local_fw_heads, L, d]
            fast_q = q_ttt.permute(0, 2, 1, 3).reshape(
                B * local_fw_heads, L, self.fw_head_dim)
            fast_k = k_ttt.permute(0, 2, 1, 3).reshape(
                B * local_fw_heads, L, self.fw_head_dim)
            fast_v = v_ttt.permute(0, 2, 1, 3).reshape(
                B * local_fw_heads, L, self.fw_head_dim)
            fast_q = l2_norm(fast_q)
            fast_k = l2_norm(fast_k)

            # lr: compute on local shard, then all-to-all
            lr = self.lr_proj(hidden_states)  # [B, L_local, 3*num_fw_heads]
            lr = lr.view(B, L_local, self.num_fw_heads, 3)
            lr = all_to_all(lr, scatter_dim=2, gather_dim=1)  # [B, L, local_fw_heads, 3]
            lr = tF.softplus(
                lr.float().permute(0, 2, 1, 3).reshape(B * local_fw_heads, L, 3)
                + self.base_lr_inv
            )
            lr0, lr1, lr2 = lr.chunk(3, dim=-1)

            # momentum
            if self.use_momentum:
                momentum = self.momentum_proj(hidden_states)  # [B, L_local, num_fw_heads]
                momentum = momentum.view(B, L_local, self.num_fw_heads, 1)
                momentum = all_to_all(momentum, scatter_dim=2, gather_dim=1)
                momentum = momentum.float().permute(0, 2, 1, 3).reshape(
                    B * local_fw_heads, L, 1)
            else:
                momentum = None

            active_fw_heads = local_fw_heads
            head_slice = slice(sp_rank * local_fw_heads, (sp_rank + 1) * local_fw_heads)
        else:
            fast_q, fast_k, fast_v = self._prepare_qkv(q, k, v)
            lr0, lr1, lr2, momentum = self._compute_lr_and_momentum(hidden_states)
            L = L_local
            active_fw_heads = self.num_fw_heads
            head_slice = slice(None)

        # ==================================================================
        # Step 2: Run TTT kernel
        # ==================================================================
        if ttt_state is None:
            # Training: initialize fresh fast weights (sliced for SP)
            w0, w1, w2 = self._get_base_weights()
            fw_w0 = w0[head_slice].repeat(B, 1, 1)
            fw_w1 = w1[head_slice].repeat(B, 1, 1)
            fw_w2 = w2[head_slice].repeat(B, 1, 1)
            if self.fp32_states:
                fw_w0, fw_w1, fw_w2 = fw_w0.float(), fw_w1.float(), fw_w2.float()

            ttt_kernel = prenorm_block_causal_lact_swiglu if self.prenorm \
                else block_causal_lact_swiglu
            fw_x = ttt_kernel(
                fw_w0, fw_w1, fw_w2,
                fast_q, fast_k, fast_v,
                lr0, lr1, lr2,
                chunk_size=self.ttt_chunk_size,
                use_muon=self.use_muon,
                momentum=momentum,
            )
        else:
            # Inference: use carried state (already sliced for SP in `init_state`)
            fw_w0 = ttt_state["w0"]
            fw_w1 = ttt_state["w1"]
            fw_w2 = ttt_state["w2"]

            # Apply: compute output with current fast weights
            fw_x = ttt_apply_only(fw_w0, fw_w1, fw_w2, fast_q)

            # Update state in-place
            (new_w0, new_w1, new_w2,
             new_dw0, new_dw1, new_dw2,
             new_w0_main, new_w1_main, new_w2_main) = ttt_update_state(
                fw_w0, fw_w1, fw_w2,
                fast_k, fast_v,
                lr0, lr1, lr2,
                use_muon=self.use_muon,
                momentum=momentum,
                dw0_momentum=ttt_state.get("dw0_momentum"),
                dw1_momentum=ttt_state.get("dw1_momentum"),
                dw2_momentum=ttt_state.get("dw2_momentum"),
                prenorm=self.prenorm,
                w0_main=ttt_state.get("w0_main", fw_w0),
                w1_main=ttt_state.get("w1_main", fw_w1),
                w2_main=ttt_state.get("w2_main", fw_w2),
                w0_norm=ttt_state["w0_norm"],
                w1_norm=ttt_state["w1_norm"],
                w2_norm=ttt_state["w2_norm"],
            )
            ttt_state["w0"] = new_w0
            ttt_state["w1"] = new_w1
            ttt_state["w2"] = new_w2
            ttt_state["w0_main"] = new_w0_main
            ttt_state["w1_main"] = new_w1_main
            ttt_state["w2_main"] = new_w2_main
            if self.use_momentum:
                ttt_state["dw0_momentum"] = new_dw0
                ttt_state["dw1_momentum"] = new_dw1
                ttt_state["dw2_momentum"] = new_dw2

        # ==================================================================
        # Step 3: Norm + Scale, then reverse SP if needed
        # ==================================================================
        if use_sp:
            # [B*local_fw_heads, L, fw_head_dim]
            ttt_x_normed = self.ttt_norm(fw_x)

            # Compute scale on local shard, then all-to-all
            scale = tF.silu(self.ttt_scale_proj(hidden_states))  # [B, L_local, num_fw_heads]
            scale = scale.view(B, L_local, self.num_fw_heads, 1)
            scale = all_to_all(scale, scatter_dim=2, gather_dim=1)  # [B, L, local_fw_heads, 1]
            scale = scale.permute(0, 2, 1, 3).reshape(B * local_fw_heads, L, 1)
            ttt_x_normed = ttt_x_normed * scale

            # Reverse all-to-all: [B, L, local_fw_heads, d] -> [B, L_local, num_fw_heads, d]
            ttt_output = ttt_x_normed.view(
                B, local_fw_heads, L, self.fw_head_dim).permute(0, 2, 1, 3)
            ttt_output = all_to_all(ttt_output, scatter_dim=1, gather_dim=2)
            ttt_output = ttt_output.flatten(2)  # [B, L_local, ttt_dim]
        else:
            ttt_output = self._apply_scale_and_norm(fw_x, hidden_states)

        # Zero-pad to full `dim`
        L_out = ttt_output.shape[1]
        if self.ttt_dim < self.dim:
            pad = torch.zeros(B, L_out, self.dim - self.ttt_dim,
                              device=ttt_output.device, dtype=ttt_output.dtype)
            ttt_output = torch.cat([ttt_output, pad], dim=-1)

        return ttt_output

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Initialize TTT state for inference (called once before autoregressive generation).
        With SP, only initializes state for this rank's local heads (`num_fw_heads // sp_size`).
        """
        sp_size = get_sp_world_size()
        sp_rank = get_sp_rank()

        w0, w1, w2 = self._get_base_weights()

        # Slice local heads for this SP rank
        if sp_size > 1:
            assert self.num_fw_heads % sp_size == 0
            local_fw_heads = self.num_fw_heads // sp_size
            h_s = sp_rank * local_fw_heads
            h_e = h_s + local_fw_heads
            w0, w1, w2 = w0[h_s:h_e], w1[h_s:h_e], w2[h_s:h_e]

        fw_w0 = w0.repeat(batch_size, 1, 1).to(device=device)
        fw_w1 = w1.repeat(batch_size, 1, 1).to(device=device)
        fw_w2 = w2.repeat(batch_size, 1, 1).to(device=device)
        if self.fp32_states:
            fw_w0 = fw_w0.float()
            fw_w1 = fw_w1.float()
            fw_w2 = fw_w2.float()

        state = {
            "w0": fw_w0, "w1": fw_w1, "w2": fw_w2,
            "w0_main": fw_w0.clone(), "w1_main": fw_w1.clone(), "w2_main": fw_w2.clone(),
            "w0_norm": fw_w0.norm(dim=2, keepdim=True),
            "w1_norm": fw_w1.norm(dim=2, keepdim=True),
            "w2_norm": fw_w2.norm(dim=2, keepdim=True),
        }
        if self.use_momentum:
            state["dw0_momentum"] = torch.zeros_like(fw_w0)
            state["dw1_momentum"] = torch.zeros_like(fw_w1)
            state["dw2_momentum"] = torch.zeros_like(fw_w2)

        return state
