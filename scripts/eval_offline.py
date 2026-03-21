"""Offline evaluation script with WandB sync.

Loads a checkpoint from a training experiment and runs evaluation independently,
then syncs results back to the training run's WandB dashboard.

Usage:
    # Single GPU eval
    python scripts/eval_offline.py --tag my_experiment --step 1000

    # Multi-GPU + SP eval
    torchrun --nproc_per_node=4 scripts/eval_offline.py \
        --tag my_experiment --step 1000 --sp_size 4

    # Without WandB logging
    python scripts/eval_offline.py --tag my_experiment --step 1000 --no_wandb
"""

import warnings
warnings.filterwarnings("ignore")
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()

import os
import sys
import argparse
import gc
import glob

from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.options import Options, ROOT
from src.data import *  # import all dataset classes
from src.models.modules import WanVAEWrapper
from src.models import Wan
import src.utils.util as util
import src.utils.vis_util as vis_util
from src.utils.distributed import fsdp_wrap, launch_distributed_job, barrier, initialize_sequence_parallel


def find_wandb_run_id(exp_dir: str) -> str:
    """Find the WandB run ID from the experiment directory.

    Looks for `wandb/latest-run/run-<run_id>.wandb` files.
    """
    pattern = os.path.join(exp_dir, "wandb", "latest-run", "run-*.wandb")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"Could not find WandB run files in {os.path.join(exp_dir, 'wandb', 'latest-run')}. "
            f"Please specify --wandb_run_id manually."
        )
    # Extract run ID from filename like `run-abc123.wandb`
    filename = os.path.basename(matches[0])
    run_id = filename.replace("run-", "").replace(".wandb", "")
    return run_id


def load_opt_from_params(exp_dir: str) -> Options:
    """Load `Options` directly from the experiment's `params.yaml`."""
    params_path = os.path.join(exp_dir, "params.yaml")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.yaml not found at {params_path}")

    params = OmegaConf.load(params_path)
    opt = Options()
    for f in opt.__dataclass_fields__:
        if f in params:
            val = params[f]
            if OmegaConf.is_missing(params, f):
                continue
            val = OmegaConf.to_container(val) if OmegaConf.is_config(val) else val
            setattr(opt, f, val)
    opt.__post_init__()
    return opt


def main():
    PROJECT_NAME = "WanCameraControl"

    parser = argparse.ArgumentParser(description="Offline evaluation with WandB sync.")
    parser.add_argument("--tag", type=str, required=True,
                        help="Experiment tag (e.g., my_experiment)")
    parser.add_argument("--step", type=int, required=True,
                        help="Training step for this checkpoint (used as wandb x-axis and sampler seed)")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="WandB run ID (auto-detected from exp_dir if not provided)")
    parser.add_argument("--max_val_steps", type=int, default=2,
                        help="Maximum number of validation steps")
    parser.add_argument("--wandb_token_path", type=str, default=f"{ROOT}/.cache/wandb/token",
                        help="Path to WandB login token")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip WandB logging, only run evaluation")
    parser.add_argument("--sp_size", type=int, default=1,
                        help="Sequence parallel size (1=single GPU, >1 requires torchrun)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloader workers")
    # parser.add_argument("--allow_tf32", action="store_true",
    #                     help="Enable TF32 for faster inference")
    args = parser.parse_args()

    # Derive paths from tag and step
    args.exp_dir = os.path.join("out", args.tag)
    args.ckpt_path = os.path.join(args.exp_dir, "checkpoints", f"{args.step:06d}", "model_states.pth")

    eval_dir = os.path.join(args.exp_dir, f"eval_{args.step:06d}")
    os.makedirs(eval_dir, exist_ok=True)

    # Enable TF32
    if True:#args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load opt from params.yaml
    opt = load_opt_from_params(args.exp_dir)

    # Override SP size from CLI
    opt.sp_size = args.sp_size

    # Distributed setup
    use_distributed = args.sp_size > 1
    if use_distributed:
        launch_distributed_job()
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size % args.sp_size == 0
        dp_size = world_size // args.sp_size
        dp_rank = global_rank // args.sp_size
    else:
        global_rank = 0
        world_size = 1
        dp_size = 1
        dp_rank = 0

    is_main_process = (global_rank == 0)
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    # Set random seed (same logic as training)
    util.set_seed(args.seed + dp_rank)

    if is_main_process:
        print(f"=== Offline Evaluation ===")
        print(f"Experiment: {args.exp_dir}")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Step: {args.step}")
        print(f"SP size: {args.sp_size}")
        print(f"Distributed: {use_distributed}")
        print()

    # Create validation dataset and dataloader
    if opt.use_internal_dataset:
        val_dataset = InternalDataset(opt, training=False)
    else:
        val_dataset = RealcamvidDataset(opt, training=False)

    if use_distributed:
        val_sampler = DistributedSampler(val_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, drop_last=False)
    else:
        val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=0, shuffle=True, drop_last=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=1 if args.num_workers > 0 else None,
        multiprocessing_context="forkserver",
        persistent_workers=args.num_workers > 0,
        collate_fn=BaseDataset.collate_fn,
    )

    if is_main_process:
        print(f"Validation dataset: {len(val_dataset)} samples")

    # Initialize the model
    model = Wan(opt)

    # Initialize sequence parallelism and FSDP if distributed
    if use_distributed:
        assert args.sp_size <= world_size and world_size % args.sp_size == 0
        assert model.diffusion.model.num_heads % args.sp_size == 0
        initialize_sequence_parallel(args.sp_size)
        if is_main_process:
            print(f"Sequence Parallelism initialized: sp_size={args.sp_size}")

        # Load checkpoint on rank 0 before FSDP wrap (`sync_module_states=True` syncs to other ranks)
        if is_main_process:
            print(f"Loading checkpoint: {args.ckpt_path}")
            state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
            model.diffusion.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()

        # FSDP wrap with `no_shard` for eval (no backward needed)
        model.diffusion = fsdp_wrap(
            model.diffusion,
            sharding_strategy="no_shard",
            mixed_precision="bf16",
        )
        model.text_encoder = fsdp_wrap(
            model.text_encoder,
            sharding_strategy="no_shard",
            mixed_precision="bf16",
        )
    else:
        # Single GPU: direct load
        if is_main_process:
            print(f"Loading checkpoint: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
        model.diffusion.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()

        model.diffusion = model.diffusion.to(device=device, dtype=dtype)
        model.text_encoder = model.text_encoder.to(device=device, dtype=dtype)

    # Prepare VAE
    vae = WanVAEWrapper(opt.vae_path)
    vae = vae.to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    if model.current_vae_decoder is not None:
        model.current_vae_decoder = model.current_vae_decoder.to(device=device, dtype=dtype)
    if model.lpips_loss is not None:
        model.lpips_loss = model.lpips_loss.to(device=device, dtype=dtype)

    # Run evaluation
    with torch.inference_mode():
        model.eval()

        all_val_outputs, all_val_metrics, val_steps = [], {}, 0
        # Use step as sampler epoch (same as training: `val_sampler.set_epoch(global_update_step)`)
        val_sampler.set_epoch(args.step)

        val_progress_bar = tqdm(
            range(len(val_loader)) if args.max_val_steps is None else range(args.max_val_steps),
            desc="Validation",
            ncols=125,
            disable=not is_main_process,
        )

        for val_batch in val_loader:
            val_outputs_gpu = model(val_batch, func_name="evaluate", vae=vae)

            # Move outputs to CPU to save GPU memory
            val_outputs = {}
            for k, v in val_outputs_gpu.items():
                if torch.is_tensor(v):
                    val_outputs[k] = v.detach().cpu()
                else:
                    val_outputs[k] = v
            all_val_outputs.append(val_outputs)

            # Collect metrics
            for cfg_scale in opt.cfg_scale:
                for metric_name in ["psnr", "ssim", "depth", "ray", "pose"]:
                    key = f"{metric_name}_{cfg_scale}"
                    if key in val_outputs:
                        all_val_metrics.setdefault(key, []).append(val_outputs[key])

            val_progress_bar.update(1)
            val_steps += 1
            del val_outputs_gpu

            if args.max_val_steps is not None and val_steps == args.max_val_steps:
                break

        val_progress_bar.close()

    if use_distributed:
        barrier()

    # Aggregate metrics
    for k, v in all_val_metrics.items():
        all_val_metrics[k] = torch.tensor(v).mean()

    # Print metrics
    if is_main_process:
        print(f"\n=== Evaluation Results (step {args.step}) ===")
        for cfg_scale in opt.cfg_scale:
            metrics_str = []
            for metric_name in ["psnr", "ssim", "depth", "ray", "pose"]:
                key = f"{metric_name}_{cfg_scale}"
                if key in all_val_metrics:
                    metrics_str.append(f"{key}: {all_val_metrics[key].item():.4f}")
            if metrics_str:
                print("  " + ", ".join(metrics_str))

    # Concatenate image outputs for visualization
    val_outputs = {}
    if all_val_outputs:
        for k in all_val_outputs[0].keys():
            if "images" in k and all_val_outputs[0][k] is not None:
                val_outputs[k] = torch.cat([out[k] for out in all_val_outputs], dim=0)

    # WandB logging
    if is_main_process and not args.no_wandb:
        import wandb

        # Resolve WandB run ID
        run_id = args.wandb_run_id
        if run_id is None:
            run_id = find_wandb_run_id(args.exp_dir)
        print(f"\nSyncing to WandB run: {run_id}")

        # Set proxy for network access
        os.environ["http_proxy"] = "http://star-proxy.oa.com:3128"
        os.environ["https_proxy"] = "http://star-proxy.oa.com:3128"

        # Login
        with open(args.wandb_token_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()

        # Resume the training run
        wandb.init(
            project=PROJECT_NAME,
            id=run_id,
            dir=eval_dir,
            resume="allow",
        )

        # Log metrics
        for cfg_scale in opt.cfg_scale:
            for metric_name in ["psnr", "ssim", "depth", "ray", "pose"]:
                key = f"{metric_name}_{cfg_scale}"
                if key in all_val_metrics:
                    wandb.log({
                        f"validation/{key}": all_val_metrics[key].item(),
                    }, step=args.step)

        # Log videos
        wandb.log({
            "videos/validation": vis_util.wandb_video_log(
                val_outputs, max_res=512, fps=16),
        }, step=args.step)

        wandb.finish()
        print("WandB sync complete.")

    # Save videos locally
    if is_main_process:
        import imageio
        for k, tensor in val_outputs.items():
            if "images" not in k or tensor is None:
                continue
            # tensor: (B, F, 3, H, W) in [0, 1]
            video_np = vis_util.tensor_to_video(tensor)  # (F, H, B*W, C) uint8
            save_path = os.path.join(eval_dir, f"{k}.mp4")
            imageio.mimwrite(save_path, video_np, fps=16)
            print(f"Saved video to {save_path}")

    # Cleanup
    del all_val_outputs, val_outputs
    torch.cuda.empty_cache()
    gc.collect()

    if is_main_process:
        print("\nDone!")


if __name__ == "__main__":
    main()
