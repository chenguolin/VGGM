# Copied from https://github.com/guandeh17/Self-Forcing/blob/main/utils/distributed.py

# Modified:
    ## 1. Reformat code style
    ## 2. Delete EMA_FSDP
    ## 3. Support sequence parallelism

from typing import *
from torch import Tensor

import os
from datetime import timedelta
from functools import partial

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model: nn.Module):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        checkpoint = model.state_dict()
    return checkpoint


def fsdp_wrap(
    module: nn.Module,
    sharding_strategy: Literal["full", "hybrid_full", "hybrid_zero2", "no_shard"] = "hybrid_full",
    mixed_precision: bool = False,
    wrap_strategy: Literal["size", "transformer"] = "size",
    min_num_params: int = int(5e7),
    transformer_module: Optional[Set[Type[nn.Module]]] = None,
    cpu_offload: bool = False,
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,  # hard-coded to bf16
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module,
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)


# ============================================================================
# Sequence Parallelism Utilities
# ============================================================================
#
# Sequence Parallelism (SP) splits the sequence dimension across multiple GPUs
# to reduce memory usage and enable longer sequences. This is complementary to
# data parallelism and model parallelism.
#
# Key operations:
# - all_gather: Gather tensors from all ranks (forward), reduce_scatter (backward)
# - all_split: Split tensor to ranks (forward), all_gather gradients (backward)
# - all_to_all: Scatter one dim and gather another (Ulysses-style attention)
#
# Example with sp_size=2:
#   Input sequence length L=1000
#   - Rank 0 processes tokens [0:500]
#   - Rank 1 processes tokens [500:1000]
#   - Communication happens via all_gather/all_to_all during attention
#
# ============================================================================

# Global variables for SP process groups
_SP_GROUP = None
_SP_RANK = None
_SP_WORLD_SIZE = None


def initialize_sequence_parallel(sp_size: int):
    """
    Initialize sequence parallel process groups.

    Args:
        sp_size: Sequence parallel size (number of ranks per SP group)

    Example:
        With world_size=64 and sp_size=2:
        - Creates 32 SP groups: [0,1], [2,3], ..., [62,63]
        - Each group has 2 ranks for sequence parallelism
        - Remaining dimension is data parallelism (32 DP groups)
    """
    global _SP_GROUP, _SP_RANK, _SP_WORLD_SIZE

    assert dist.is_initialized()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Calculate which SP group this rank belongs to
    num_sp_groups = world_size // sp_size
    sp_group_idx = rank // sp_size
    sp_rank_in_group = rank % sp_size

    # Create SP process groups
    # Each group contains sp_size consecutive ranks
    for i in range(num_sp_groups):
        ranks_in_group = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks_in_group)
        if rank in ranks_in_group:
            _SP_GROUP = group
            _SP_RANK = sp_rank_in_group
            _SP_WORLD_SIZE = sp_size

    print(f"[Rank {rank}] Initialized SP: sp_group_idx={sp_group_idx}, "
          f"sp_rank={_SP_RANK}, sp_world_size={_SP_WORLD_SIZE}")

    # Barrier to ensure all ranks have finished creating SP groups
    barrier()


def get_sp_rank():
    """Get sequence parallel rank within the SP group."""
    if _SP_RANK is not None:
        return _SP_RANK
    return 0


def get_sp_world_size():
    """Get sequence parallel world size (size of SP group)."""
    if _SP_WORLD_SIZE is not None:
        return _SP_WORLD_SIZE
    return 1


def get_sp_group():
    """Get the sequence parallel process group."""
    return _SP_GROUP


def _resolve_group_info(group=None):
    if group is None:
        return get_sp_group(), get_sp_world_size(), get_sp_rank()
    return group, dist.get_world_size(group=group), dist.get_rank(group=group)


def _all_to_all_impl(x: Tensor, scatter_dim: int, gather_dim: int, group, world_size: int, **kwargs):
    if x.size(scatter_dim) % world_size != 0:
        raise ValueError(
            f"all_to_all requires x.size({scatter_dim}) divisible by world_size, "
            f"got {x.size(scatter_dim)} and {world_size}"
        )
    inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(u) for u in inputs]
    dist.all_to_all(outputs, inputs, group=group, **kwargs)
    return torch.cat(outputs, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """
    All-to-all communication primitive for sequence parallelism.

    Scatters along one dimension and gathers along another dimension.
    Used in Ulysses-style distributed attention to switch between
    sequence-parallel and head-parallel layouts.

    Example:
        Input:  [B, L//sp, N, C] (sequence split, full heads)
        Output: [B, L, N//sp, C] (full sequence, heads split)
    """

    @staticmethod
    def forward(ctx, x: Tensor, scatter_dim: int, gather_dim: int, group):
        group, world_size, _ = _resolve_group_info(group)
        if world_size <= 1:
            ctx.group = None
            ctx.world_size = 1
            return x

        scatter_dim = scatter_dim % x.dim()
        gather_dim = gather_dim % x.dim()
        ctx.group = group
        ctx.world_size = world_size
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all_impl(x, scatter_dim, gather_dim, group, world_size)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass: reverse the all-to-all by swapping scatter and gather dims.
        """
        if ctx.world_size <= 1:
            return grad_output, None, None, None
        grad_input = _all_to_all_impl(
            grad_output,
            scatter_dim=ctx.gather_dim,
            gather_dim=ctx.scatter_dim,
            group=ctx.group,
            world_size=ctx.world_size,
        )
        return grad_input, None, None, None


class _AllGather(torch.autograd.Function):
    """
    All-gather communication primitive for sequence parallelism.

    Forward: Gather tensors from all ranks along specified dimension
    Backward: Reduce-scatter gradients back to each rank

    This is used to collect distributed sequences before operations that
    require the full sequence (e.g., certain attention patterns).

    Example with sp_size=2:
        Forward:
            Rank 0: [B, L//2, C] -> [B, L, C]
            Rank 1: [B, L//2, C] -> [B, L, C]
        Backward:
            Each rank receives sum of gradients for its portion
    """

    @staticmethod
    def forward(ctx, input: Tensor, dim: int, group):
        group, world_size, rank = _resolve_group_info(group)
        if world_size <= 1:
            ctx.group = None
            ctx.world_size = 1
            return input

        dim = dim % input.dim()
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.dim = dim

        tensor_list = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(tensor_list, input, group=group)
        return torch.cat(tensor_list, dim=dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward: Use reduce_scatter to sum gradients and distribute to ranks.

        Each rank contributed a chunk to the gathered output, so gradients
        for each chunk need to be summed across all ranks (since all ranks
        see the full output and compute gradients for all positions).
        """
        if ctx.world_size <= 1:
            return grad_output, None, None

        if grad_output.size(ctx.dim) % ctx.world_size != 0:
            raise ValueError(
                f"all_gather backward requires grad_output.size({ctx.dim}) divisible by world_size, "
                f"got {grad_output.size(ctx.dim)} and {ctx.world_size}"
            )
        grad_output_chunks = [u.contiguous() for u in torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)]
        grad_input = torch.empty_like(grad_output_chunks[0])
        dist.reduce_scatter(grad_input, grad_output_chunks, group=ctx.group)
        return grad_input, None, None


class _AllSplit(torch.autograd.Function):
    """
    Split (scatter) communication primitive for sequence parallelism.

    Forward: Split tensor and return this rank's chunk
    Backward: All-gather gradients from all ranks

    This is the inverse of all_gather:
    - all_gather: forward gathers, backward reduces and scatters
    - all_split: forward splits, backward gathers

    Used to distribute sequences at the start of SP regions.

    Example with sp_size=2:
        Forward:
            Input: [B, L, C] -> Rank 0: [B, L//2, C], Rank 1: [B, L//2, C]
        Backward:
            Each rank's gradient [B, L//2, C] -> gathered to [B, L, C]
    """

    @staticmethod
    def forward(ctx, input: Tensor, dim: int, group):
        group, world_size, rank = _resolve_group_info(group)
        if world_size <= 1:
            ctx.group = None
            ctx.world_size = 1
            return input

        if input.size(dim) % world_size != 0:
            raise ValueError(
                f"split requires input.size({dim}) divisible by world_size, "
                f"got {input.size(dim)} and {world_size}"
            )

        dim = dim % input.dim()
        ctx.group = group
        ctx.world_size = world_size
        ctx.dim = dim

        # Forward: split and take this rank's chunk
        return torch.chunk(input, world_size, dim=dim)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward: All-gather to collect gradients from all ranks.

        Since forward split the input, backward needs to gather the gradients
        back together. No reduction is needed because each rank only computed
        gradients for its own chunk.
        """
        if ctx.world_size <= 1:
            return grad_output, None, None

        # Backward: all_gather to collect gradients from all ranks
        tensor_list = [torch.empty_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list, grad_output, group=ctx.group)
        grad_input = torch.cat(tensor_list, dim=ctx.dim).contiguous()
        return grad_input, None, None


def all_to_all(x: Tensor, scatter_dim: int, gather_dim: int, group=None):
    """
    Scatter along one dimension and gather along another.

    Args:
        x: Input tensor
        scatter_dim: Dimension to scatter (split)
        gather_dim: Dimension to gather (concatenate)
        group: Process group (default: None, uses SP group)

    Returns:
        Tensor with scatter_dim split and gather_dim concatenated
    """
    return _AllToAll.apply(x, scatter_dim, gather_dim, group)


def all_gather(input: Tensor, dim: int, group=None):
    """
    Gather tensor along specified dimension across all ranks in SP group.

    Args:
        input: Input tensor
        dim: Dimension to gather along
        group: Process group (default: None, uses SP group)

    Returns:
        Gathered tensor
    """
    return _AllGather.apply(input, dim, group)


def all_split(input: Tensor, dim: int, group=None):
    """
    Split tensor along specified dimension and take this rank's chunk.

    Forward: Split input and return this rank's portion
    Backward: All-gather gradients from all ranks

    This is the inverse of all_gather:
    - all_gather: forward gathers, backward scatters (reduce_scatter)
    - all_split: forward scatters, backward gathers (all_gather)

    Args:
        input: Input tensor
        dim: Dimension to split along
        group: Process group (default: None, uses SP group)

    Returns:
        This rank's chunk of the tensor
    """
    return _AllSplit.apply(input, dim, group)
