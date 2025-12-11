from typing import *
from torch import Tensor

import torch
from torch import nn
from einops import rearrange
from flash_attn import flash_attn_func


class TransformerRNN(nn.Module):
    def __init__(self,
        q_dim: int,
        dim: int,
        num_heads: int,
        num_state_tokens: int,
        num_blocks: int,
    ):
        super().__init__()

        self.proj_q = nn.Linear(q_dim, dim)
        self.init_state = nn.Embedding(num_state_tokens, dim)

        self.state, self.num_state_tokens = None, num_state_tokens

        self.write_layers = nn.ModuleList([Block(dim, num_heads) for _ in range(num_blocks)])
        self.read_layers = nn.ModuleList([Block(dim, num_heads) for _ in range(num_blocks)])

    def get_initial_state(self, context: Tensor):
        B = context.shape[0]
        state = self.init_state(torch.arange(self.num_state_tokens, device=context.device))
        return state.expand(B, -1, -1)

    def forward(self, q: Tensor):
        x = self.proj_q(q)

        if self.state is None:
            self.state = self.get_initial_state(q)

        for w, r in zip(self.write_layers, self.read_layers):
            self.state = w(self.state, x)
            x = r(x, self.state)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        zero_init: bool = False,
        mlp_expansion_ratio: int = 4,
    ):
        super().__init__()

        self.norm1 = nn.RMSNorm(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, proj_bias, zero_init)
        self.context_norm = nn.RMSNorm(dim)

        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_expansion_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_expansion_ratio, dim),
        )

    def forward(self, x: Tensor, context: Tensor):
        x = x + self.attn(self.norm1(x), self.context_norm(context))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        zero_init: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_o = nn.Linear(dim, dim, bias=proj_bias)

        if zero_init:
            nn.init.zeros_(self.to_o.weight)
            nn.init.zeros_(self.to_o.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        q = rearrange(self.to_q(x), "b n (h hd) -> b n h hd", h=self.num_heads)
        k = rearrange(self.to_k(context), "b m (h hd) -> b m h hd", h=self.num_heads)
        v = rearrange(self.to_v(context), "b m (h hd) -> b m h hd", h=self.num_heads)

        o = rearrange(flash_attn_func(q, k, v), "b n h hd -> b n (h hd)")
        return self.to_o(o)
