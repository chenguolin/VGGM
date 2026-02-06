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


def get_max_grad_norm(model: Module) -> Union[str, Tensor]:
    max_name, max_grad_norm = "NONE", torch.tensor(0.)
    for (name, param) in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)  # (1,)
            if grad_norm >= max_grad_norm:
                max_grad_norm = grad_norm
                max_name = name

    if max_grad_norm.item() == 0. and max_name != "NONE":
        max_name = "ZERO"

    return max_name, max_grad_norm


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
