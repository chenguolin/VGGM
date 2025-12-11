from typing import *
from argparse import Namespace
from omegaconf import DictConfig
from torch.nn import Module
from accelerate import Accelerator
from src.options import Options

import os
import subprocess
from omegaconf import OmegaConf
from accelerate import load_checkpoint_and_dispatch


def load_ckpt(
    ckpt_dir: str, ckpt_iter: int,
    model: Optional[Module] = None,
    accelerator: Optional[Accelerator] = None,
    strict: bool = True,
    use_ema: bool = False,
) -> Module:
    # Find the latest checkpoint
    if ckpt_iter < 0:
        ckpt_iter = int(sorted(os.listdir(ckpt_dir))[-1])

    # Download checkpoint
    ckpt_path = f"{ckpt_dir}/{ckpt_iter:06d}" + ("/ema_states.pth" if use_ema else "")
    assert os.path.exists(ckpt_path)

    if model is None:
        return ckpt_iter

    # Load checkpoint
    else:
        ckpt_dir = f"{ckpt_dir}/{ckpt_iter:06d}"
        if not os.path.exists(f"{ckpt_dir}/zero_to_fp32.py"):
            load_checkpoint_and_dispatch(model, ckpt_path, strict=strict)
        else:  # from DeepSpeed
            if accelerator is not None:
                if accelerator.is_local_main_process:
                    ensure_sysrun(f"python3 {ckpt_dir}/zero_to_fp32.py {ckpt_dir} {ckpt_dir} --safe_serialization")
                accelerator.wait_for_everyone()  # wait before preparing checkpoints by the main process
            else:
                ensure_sysrun(f"python3 {ckpt_dir}/zero_to_fp32.py {ckpt_dir} {ckpt_dir} --safe_serialization")
            load_checkpoint_and_dispatch(model, ckpt_path, strict=strict)

        return model, ckpt_iter


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
