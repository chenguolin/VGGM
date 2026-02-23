from typing import *
from argparse import Namespace
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module
from src.options import Options

import os
import subprocess
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import torch.distributed as dist


def tensor_is_finite(x: Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().item())


def dist_any_true(flag: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return flag
    flag_tensor = torch.tensor([int(flag)], device=device, dtype=torch.int32)
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
    return bool(flag_tensor.item())


def find_nonfinite_grad_names(model: Module, max_names: int = 3) -> List[str]:
    nonfinite_names = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all().item():
            nonfinite_names.append(name)
            if len(nonfinite_names) >= max_names:
                break
    return nonfinite_names


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def get_configs(yaml_path: str, cli_configs: List[str]=[], **kwargs) -> DictConfig:
    yaml_configs = OmegaConf.load(yaml_path)
    cli_configs = OmegaConf.from_cli(cli_configs)

    configs = OmegaConf.merge(yaml_configs, cli_configs, kwargs)
    OmegaConf.resolve(configs)  # resolve ${...} placeholders
    return configs


def save_experiment_params(args: Namespace, configs: DictConfig, opt: Options, save_dir: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    params = OmegaConf.merge(configs, {k: str(v) for k, v in vars(args).items()})
    params = OmegaConf.merge(params, OmegaConf.create(vars(opt)))
    OmegaConf.save(params, os.path.join(save_dir, "params.yaml"))
    return dict(params)


def save_model_architecture(model: Module, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    num_buffers = sum(b.numel() for b in model.buffers())
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f"Number of buffers: {num_buffers}\n" +\
        f"Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n" +\
        f"Model architecture:\n{model}"

    with open(os.path.join(save_dir, "model.txt"), "w") as f:
        f.write(message)


def get_git_version():
    try:
        # Get current commit hash (short format)
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Check if there are any uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        dirty = bool(status)
        return commit_hash + ("-dirty" if dirty else "")

    except Exception:
        return "unknown"


def ensure_sysrun(cmd: str):
    while True:
        result = os.system(cmd)
        if result == 0:
            break
        else:
            print(f"Retry running {cmd}")
