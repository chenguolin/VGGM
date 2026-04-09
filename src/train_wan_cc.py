import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from typing import *

import os
import argparse
import logging
import math
import gc

from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import opt_dict, ROOT
from src.data import *  # import all dataset classes and `yield_forever`
from src.models.modules import WanVAEWrapper
from src.models import Wan, DMD_Wan, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.vis_util as vis_util
from src.utils.ema import EMAParams
from src.utils.distributed import fsdp_wrap, fsdp_state_dict, launch_distributed_job, barrier, initialize_sequence_parallel


def main():
    PROJECT_NAME = "WanCameraControl"

    parser = argparse.ArgumentParser(
        description="Fine-tuning Wan for camera control."
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--wandb_token_path",
        type=str,
        default=f"{ROOT}/.cache/wandb/token",
        help="Path to the WandB login token"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--skip_steps",
        type=int,
        default=0,
        help="Skip the first N training steps (fast-forward dataloader without forward pass)"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training"
    )

    parser.add_argument(
        "--sharding_strategy",
        type=str,
        choices=["full", "hybrid_full", "hybrid_zero2", "no_shard"],
        default="hybrid_full",
        help="Sharding strategy for FSDP",
    )
    parser.add_argument(
        "--wrap_strategy",
        type=str,
        choices=["size", "transformer"],
        default="size",
        help="Wrap strategy for FSDP",
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    opt.__post_init__()
    opt.git_version = util.get_git_version()

    # Create an experiment directory using the `tag`
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set args by configs
    args.gradient_accumulation_steps = max(
        args.gradient_accumulation_steps,
        configs["train"].get("gradient_accumulation_steps", 1),
    )
    if args.gradient_accumulation_steps != 1:
        raise NotImplementedError  # TODO: support gradient accumulation
    args.sharding_strategy = configs["train"].get("sharding_strategy", args.sharding_strategy)
    args.wrap_strategy = configs["train"].get("wrap_strategy", args.wrap_strategy)
    args.mixed_precision = configs["train"].get("mixed_precision", args.mixed_precision)
    args.use_ema = args.use_ema or configs["train"].get("use_ema", False)

    # Enable TF32 for faster training
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Distribution setting
    launch_distributed_job()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = {"no": torch.float32,
        "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    device = torch.cuda.current_device()
    is_main_process = (global_rank == 0)
    dp_size = world_size // opt.sp_size
    dp_rank = global_rank // opt.sp_size

    # Initialize the logger
    logger = logging.getLogger(__name__)
    if is_main_process:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S"
        )
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # File handler
        fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        if not logger.handlers:  # avoid duplicate handlers
            logger.addHandler(ch)
            logger.addHandler(fh)
        logger.propagate = False  # not propagate to the root logger (console)
    else:
        logger.disabled = True

    # Set the random seed
    if args.seed < 0:
        random_seed = torch.randint(0, 10000000, (1,), device=device)
        dist.broadcast(random_seed, src=0)
        args.seed = random_seed.item()
    logger.info(f"Random seed: [{args.seed}]\n")
    # Keep RNG streams identical within each SP group (same dp_rank)
    # while preserving different streams across DP groups
    util.set_seed(args.seed + dp_rank)  # util.set_seed(args.seed + global_rank)

    # Load train and validation datasets
    if opt.version_new_action:
        train_dataset = InternalActionDataset(opt, training=True)
    elif opt.use_internal_dataset:
        train_dataset = InternalDataset(opt, training=True)
    else:
        train_dataset = RealcamvidDataset(opt, training=True)
    # SP-aware sampling: ranks in the same SP group should get the same samples
    train_sampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=args.seed, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=1 if args.num_workers > 0 else None,  # to save CPU memory
        multiprocessing_context="forkserver",  # to save CPU memory
        persistent_workers=args.num_workers > 0,
        collate_fn=BaseDataset.collate_fn,
    )
    if opt.version_new_action:
        val_dataset = InternalActionDataset(opt, training=False)
    elif opt.use_internal_dataset:
        val_dataset = InternalDataset(opt, training=False)
    else:
        val_dataset = RealcamvidDataset(opt, training=False)
    # SP-aware sampling: ranks in the same SP group should get the same samples
    val_sampler = DistributedSampler(val_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True, seed=args.seed, drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=1 if args.num_workers > 0 else None,  # to save CPU memory
        multiprocessing_context="forkserver",  # to save CPU memory
        persistent_workers=args.num_workers > 0,
        collate_fn=BaseDataset.collate_fn,
    )
    logger.info(f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Initialize the model
    lazy = (global_rank != 0)  # only rank 0 loads state dicts; FSDP `sync_module_states` broadcasts
    if opt.use_dmd:
        model = DMD_Wan(opt, lazy=lazy)
    else:
        model = Wan(opt, lazy=lazy)

    logger.info(f"Trainable parameter names: {sorted([name for name, param in model.named_parameters() if param.requires_grad])}\n")
    if is_main_process:  # save model architecture
        util.save_model_architecture(model, exp_dir)

    # Initialize sequence parallelism
    if opt.sp_size > 1:
        assert opt.sp_size <= world_size and world_size % opt.sp_size == 0
        assert model.diffusion.model.num_heads % opt.sp_size == 0
        initialize_sequence_parallel(opt.sp_size)  # set some global variables
        logger.info(f"Sequence Parallelism initialized: sp_size=[{opt.sp_size}], world_size=[{world_size}], num_heads=[{model.diffusion.model.num_heads}]")
        logger.info(f"Data parallel groups: [{world_size // opt.sp_size}], SP ranks per group: [{opt.sp_size}]\n")

    # FSDP wrap
    if args.wrap_strategy == "transformer":
        from src.models.modules.wan_modules.t5 import T5Attention
        from src.models.modules.wan_modules.model import WanAttentionBlock
        from src.models.modules.wan_modules.causal_model import CausalWanAttentionBlock
        transformer_blocks = {T5Attention, WanAttentionBlock, CausalWanAttentionBlock}
        if opt.load_da3:
            from depth_anything_3.model.dinov2.layers.block import Block
            transformer_blocks.add(Block)
    else:
        transformer_blocks = None
    model.diffusion = fsdp_wrap(
        model.diffusion,
        sharding_strategy=args.sharding_strategy,
        wrap_strategy=args.wrap_strategy,
        transformer_module=transformer_blocks,
        mixed_precision=args.mixed_precision,
    )
    model.text_encoder = fsdp_wrap(
        model.text_encoder,
        sharding_strategy=args.sharding_strategy,
        wrap_strategy=args.wrap_strategy,
        transformer_module=transformer_blocks,
        mixed_precision=args.mixed_precision,
    )
    if opt.use_dmd:
        model.real_score = fsdp_wrap(
            model.real_score,
            sharding_strategy=args.sharding_strategy,
            wrap_strategy=args.wrap_strategy,
            transformer_module=transformer_blocks,
            mixed_precision=args.mixed_precision,
            cpu_offload=opt.real_score_offload,
        )
        model.fake_score = fsdp_wrap(
            model.fake_score,
            sharding_strategy=args.sharding_strategy,
            wrap_strategy=args.wrap_strategy,
            transformer_module=transformer_blocks,
            mixed_precision=args.mixed_precision,
        )

    # Prepare VAE and optionally other modules
    vae = WanVAEWrapper(opt.vae_path)
    vae = vae.to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    if model.current_vae_decoder is not None:
        model.current_vae_decoder = model.current_vae_decoder.to(device=device, dtype=dtype)
    if model.lpips_loss is not None:
        model.lpips_loss = model.lpips_loss.to(device=device, dtype=dtype)

    # Initialize the optimizer
    if opt.name_lr_mult is not None or opt.exclude_name_lr_mult is not None:
        name_params, name_params_lr_mult = {}, {}
        for name, param in model.named_parameters():
            # Include
            if opt.name_lr_mult is not None:
                assert opt.exclude_name_lr_mult is None
                for k in opt.name_lr_mult.split(","):
                    if k in name:
                        name_params_lr_mult[name] = param
            if opt.name_lr_mult is not None and name not in name_params_lr_mult:
                name_params[name] = param
            # Exclude
            if opt.exclude_name_lr_mult is not None:
                assert opt.name_lr_mult is None
                for k in opt.exclude_name_lr_mult.split(","):
                    if k in name:
                        name_params[name] = param
            if opt.exclude_name_lr_mult is not None and name not in name_params:
                name_params_lr_mult[name] = param
        optimizer = get_optimizer(
            params=[
                # Sorted names to ensure the same order for optimizer resuming
                {"params": list([name_params[name] for name in sorted(name_params.keys())]), "lr": configs["optimizer"]["lr"]},
                {"params": list([name_params_lr_mult[name] for name in sorted(name_params_lr_mult.keys())]), "lr": configs["optimizer"]["lr"] * opt.lr_mult}
            ],
            **configs["optimizer"]
        )
        if opt.exclude_name_lr_mult is not None:
            logger.info(f"Learning rate x [1.0] parameter names: {sorted(name_params.keys())}\n")
        else:
            logger.info(f"Learning rate x [{opt.lr_mult}] parameter names: {sorted(name_params_lr_mult.keys())}\n")
    else:
        name_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
        optimizer = get_optimizer(
            # Sorted names to ensure the same order for optimizer resuming
            params=list([name_params[name] for name in sorted(name_params.keys())]),
            **configs["optimizer"]
        )

    # Initialize the learning rate scheduler
    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) / args.gradient_accumulation_steps)  # only account updated steps
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])

    # Initialize the EMA model
    ema_params = None
    if args.use_ema:
        ema_weight = configs["train"].get("ema_weight", 0.)
        if ema_weight > 0.:
            name_to_trainable_params = {}
            for name, param in model.diffusion.named_parameters():
                if not param.requires_grad:
                    continue
                renamed_name = name.replace("_fsdp_wrapped_module.", "") \
                   .replace("_checkpoint_wrapped_module.", "") \
                   .replace("_orig_mod.", "") \
                   .replace("module.", "")
                name_to_trainable_params[renamed_name] = param
            ema_params = EMAParams(name_to_trainable_params, ema_weight)
            num_ema_params = sum(p.numel() for p in ema_params.name_to_ema_params.values())
            logger.info("Set up EMA for trainable params in the diffusion model")
            logger.info(f"Number of (sharded) EMA parameters: [{(num_ema_params / 1e6):.2f} M]\tEMA weight: [{ema_weight}]\n")

    # Training config summary
    configs["train"]["total_batch_size"] = configs["train"]["batch_size_per_gpu"] * dp_size * args.gradient_accumulation_steps
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]  # configs["train"]["epochs"] * updated_steps_per_epoch
    logger.info("===== Training Configuration Summary =====")
    logger.info(f"Total batch size: [{configs['train']['total_batch_size']}]")
    if opt.sp_size > 1:
        logger.info(f"    batch_size [{configs['train']['total_batch_size']}] = batch_size_per_gpu [{configs['train']['batch_size_per_gpu']}] * dp_size [{dp_size}] * grad_accum [{args.gradient_accumulation_steps}]")
        logger.info(f"    world_size [{world_size}] = dp_size [{dp_size}] * sp_size [{opt.sp_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{configs['lr_scheduler']['total_steps']}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps

    # Save all experimental parameters of this run to a file (args, configs, and opt)
    if is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)

    # WandB logger
    if is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        with open(args.wandb_token_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    # Set the random seed again before training loop
    util.set_seed(args.seed + dp_rank)  # util.set_seed(args.seed + global_rank)

    # Start training
    NONFINITE_SKIP_COUNT, global_update_step = 0, 0
    if is_main_process:
        logger.removeHandler(logger.handlers[0])  # remove console handler during training
    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not is_main_process
    )
    for epoch in range(configs["train"]["epochs"]):

        train_sampler.set_epoch(epoch)  # for shuffling the training dataset
        for batch in train_loader:

            # Fast-forward dataloader to `skip_steps` without running forward pass
            if global_update_step < args.skip_steps:
                global_update_step += 1
                progress_bar.update(1)
                continue

            if global_update_step == args.max_train_steps:
                progress_bar.close()
                if is_main_process:
                    wandb.finish()
                logger.info("Training finished!\n")
                return

            model.train()

            is_eval = ((global_update_step % configs["train"]["early_eval_freq"] == 0 and
                global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == args.max_train_steps-1
            )

            if opt.use_dmd:
                train_generator = global_update_step % opt.generator_train_every == 0
                outputs = model(batch, dtype=dtype, train_generator=train_generator, is_eval=is_eval, vae=vae)
            else:
                outputs = model(batch, dtype=dtype, is_eval=is_eval, vae=vae)

            loss = outputs["loss"]

            # Batch-extract optional training metrics from `outputs`
            _METRIC_KEYS = [
                "diffusion_loss", "critic_loss", "generator_loss", "dmd_grad_norm",
                "depth_loss", "depth_loss_diffusion", "ray_loss", "ray_loss_diffusion",
                "camera_loss", "camera_loss_diffusion", "render_loss",
            ]
            train_metrics = {k: outputs[k] for k in _METRIC_KEYS if k in outputs}

            # Backpropagate
            # For DMD: sequential backward passes to reduce peak activation memory
            if "losses" in outputs:
                for i, sub_loss in enumerate(outputs["losses"]):
                    sub_loss.backward()
                    if i < len(outputs["losses"]) - 1:
                        gc.collect()
                        torch.cuda.empty_cache()
            else:
                loss.backward()

            # Skip the step if any rank produces NaN/Inf gradients
            local_nonfinite_grad_names = util.find_nonfinite_grad_names(model)
            local_nonfinite_grad = len(local_nonfinite_grad_names) > 0
            any_nonfinite_grad = util.dist_any_true(local_nonfinite_grad, loss.device)
            if any_nonfinite_grad:
                logger.warning(f"Step [{global_update_step:06d}] gradients [{local_nonfinite_grad_names}] are non-finite on some rank, skip the step")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                NONFINITE_SKIP_COUNT += 1
                if NONFINITE_SKIP_COUNT > 10:
                    logger.error(f"Non-finite loss/grad skipped for [{NONFINITE_SKIP_COUNT}] consecutive steps! Training will abort.")
                    barrier()  # ensure all ranks see this error before raising
                    raise ValueError(f"Non-finite loss/grad skipped for [{NONFINITE_SKIP_COUNT}] consecutive steps!")

                barrier()  # ensure all ranks are synchronized before continuing
                continue

            # Gradient clip
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad], args.max_grad_norm)

            # Update the model parameters
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            NONFINITE_SKIP_COUNT = 0

            # Update the EMA model
            if ema_params is not None:
                if global_update_step >= configs["train"].get("ema_start_step", 0):
                    ema_params.update()
                else:  # EMA starts from current paramters
                    ema_params = EMAParams(name_to_trainable_params, ema_weight)

            # Logging
            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if "diffusion_loss" in train_metrics:
                logs["diffusion"] = train_metrics["diffusion_loss"].item()
            if "critic_loss" in train_metrics:
                logs["critic"] = train_metrics["critic_loss"].item()
            if "generator_loss" in train_metrics:
                logs["generator"] = train_metrics["generator_loss"].item()

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

            logger.info(
                f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}" +
                (f", critic: {logs['critic']:.4f}" if "critic_loss" in train_metrics else "") +
                (f", generator: {logs['generator']:.4f}" if "generator_loss" in train_metrics else "")
            )

            # WandB log the training progress
            if (global_update_step % configs["train"]["log_freq"] == 0  # 1. every `log_freq` steps
                or global_update_step % updated_steps_per_epoch == 0):  # 2. every epoch
                if is_main_process:
                    wandb_log = {
                        "training/loss": loss.item(),
                        "training/lr": lr_scheduler.get_last_lr()[0],
                    }
                    for k, v in train_metrics.items():
                        wandb_log[f"training/{k}"] = v.item()
                    wandb.log(wandb_log, step=global_update_step)

            # Save checkpoint
            if global_update_step != 0 and (global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == args.max_train_steps-1):  # 3. last step

                gc.collect()

                # Use EMA parameters for saving
                if ema_params is not None:
                    # Store the model parameters temporarily and load the EMA parameters
                    ema_params.cache_model(cpu=True)
                    ema_params.copy_to_model()
                barrier()  # make sure all processes have finished the above operations before saving checkpoints

                # NOTE: For FSDP full state dict, all ranks must participate in `state_dict()` collectives even when `rank0_only=True`
                if opt.use_lora_in_wan and opt.save_lora_only:
                    # Only save LoRA weights
                    lora_state_dict = model.get_lora_state_dict()  # all ranks must call get_lora_state_dict() for FSDP collective communication
                    if is_main_process:
                        os.makedirs(os.path.join(ckpt_dir, f"{global_update_step:06d}"), exist_ok=True)
                        torch.save(lora_state_dict, os.path.join(ckpt_dir, f"{global_update_step:06d}", "lora_weights.pth"))
                        print(f"Saved LoRA-only weights to {os.path.join(ckpt_dir, f'{global_update_step:06d}', 'lora_weights.pth')}")

                else:
                    # Save full model state dict
                    save_state_dict = fsdp_state_dict(model.diffusion)
                    if opt.use_lora_in_wan:
                        lora_state_dict = model.get_lora_state_dict()  # all ranks must call get_lora_state_dict() for FSDP collective communication
                    if is_main_process:
                        os.makedirs(os.path.join(ckpt_dir, f"{global_update_step:06d}"), exist_ok=True)
                        torch.save(save_state_dict, os.path.join(ckpt_dir, f"{global_update_step:06d}", "model_states.pth"))
                        print(f"Saved model checkpoints to {os.path.join(ckpt_dir, f'{global_update_step:06d}', 'model_states.pth')}")

                        # If using LoRA, also save LoRA weights separately for easier loading
                        if opt.use_lora_in_wan:
                            torch.save(lora_state_dict, os.path.join(ckpt_dir, f"{global_update_step:06d}", "lora_weights.pth"))
                            print(f"Saved LoRA weights to {os.path.join(ckpt_dir, f'{global_update_step:06d}', 'lora_weights.pth')}")
                    del save_state_dict
                    gc.collect()
                    torch.cuda.empty_cache()

                if ema_params is not None:
                    # Switch back to the original model parameters
                    ema_params.restore_model_from_cache()
                barrier()  # make sure all processes have finished restoring the model parameters before the next training step

                gc.collect()

            # Evaluate on the validation set
            if not opt.eval_offline and \
                ((global_update_step % configs["train"]["early_eval_freq"] == 0 and
                global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == args.max_train_steps-1):  # 4. last step

                torch.cuda.empty_cache()
                gc.collect()

                # Use EMA parameters for evaluation
                if ema_params is not None:
                    # Store the model parameters temporarily and load the EMA parameters
                    ema_params.cache_model(cpu=True)
                    ema_params.copy_to_model()
                barrier()  # make sure all processes have finished the above operations before evaluation

                with torch.inference_mode():
                    model.eval()

                    all_val_outputs, all_val_metrics, val_steps = [], {}, 0
                    val_progress_bar = tqdm(
                        range(len(val_loader)) if args.max_val_steps is None \
                            else range(args.max_val_steps),
                        desc="Validation",
                        ncols=125,
                        disable=not is_main_process
                    )
                    val_sampler.set_epoch(global_update_step)  # for shuffling the validation dataset
                    for val_batch in val_loader:
                        val_outputs_gpu = model(val_batch, func_name="evaluate", vae=vae)
                        val_outputs = {}
                        for k, v in val_outputs_gpu.items():
                            if torch.is_tensor(v):
                                val_outputs[k] = v.detach().cpu()  # to save GPU memory
                            else:
                                val_outputs[k] = v
                        all_val_outputs.append(val_outputs)

                        for cfg_scale in opt.cfg_scale:
                            for metric_name in ["psnr", "ssim", "depth", "ray", "pose"]:
                                key = f"{metric_name}_{cfg_scale}"
                                if key in val_outputs:
                                    all_val_metrics.setdefault(key, []).append(val_outputs[key])
                        val_progress_bar.update(1)
                        val_steps += 1
                        del val_outputs_gpu  # to save GPU memory

                        if args.max_val_steps is not None and val_steps == args.max_val_steps:
                            break

                val_progress_bar.close()

                if ema_params is not None:
                    # Switch back to the original model parameters
                    ema_params.restore_model_from_cache()
                barrier()  # make sure all processes have finished restoring the model parameters before the next training step

                for k, v in all_val_metrics.items():
                    all_val_metrics[k] = torch.tensor(v).mean()

                for cfg_scale in opt.cfg_scale:
                    if f"psnr_{cfg_scale}" in all_val_metrics and f"ssim_{cfg_scale}" in all_val_metrics:
                        logger.info(
                            f"Eval [{global_update_step:06d} / {total_updated_steps:06d}] " +
                            f"psnr_{cfg_scale}: {all_val_metrics[f'psnr_{cfg_scale}'].item():.4f}, " +
                            f"ssim_{cfg_scale}: {all_val_metrics[f'ssim_{cfg_scale}'].item():.4f}\n"
                        )

                # outputs = accelerator.gather(outputs)
                # val_outputs = accelerator.gather_for_metrics(val_outputs)
                for k in all_val_outputs[0].keys():
                    if "images" in k and all_val_outputs[0][k] is not None:  # for visualization
                        val_outputs[k] = torch.cat([out[k] for out in all_val_outputs], dim=0)
                # Collect prompts and optional metadata from all validation batches.
                # Each entry: (output_key, scope)
                #   scope="sample" -> shown once per sample (first clip row only, blank for rest)
                #   scope="clip"   -> shown per clip (indexed by clip j)
                _META_FIELDS = [
                    ("global_caption",   "sample"),
                    ("control_agent",    "sample"),
                    ("caption_abs",        "clip"),
                    ("caption_deltas",     "clip"),
                    ("action_labels",      "clip"),
                    ("end_states",         "clip"),
                    ("frame_ranges",       "clip"),
                ]
                all_val_prompts = []
                all_val_meta = {key: [] for key, _ in _META_FIELDS}
                for out in all_val_outputs:
                    if "prompts" in out:
                        all_val_prompts.extend(out["prompts"])
                    for key, _ in _META_FIELDS:
                        if key in out:
                            all_val_meta[key].extend(out[key])
                # Keep only fields that have data matching `all_val_prompts` in length
                active_fields = [
                    (key, scope) for key, scope in _META_FIELDS
                    if len(all_val_meta[key]) == len(all_val_prompts)
                ]

                if is_main_process:
                    # Log evaluation metrics and videos
                    val_wandb_log = {f"validation/{k}": v for k, v in all_val_metrics.items()}
                    val_wandb_log["videos/training"] = vis_util.wandb_video_log(
                        outputs, max_res=512, fps=16)  # resize videos to `512` for logging
                    val_wandb_log["videos/validation"] = vis_util.wandb_video_log(
                        val_outputs, max_res=512, fps=16)  # resize videos to `512` for logging
                    # Log evaluation captions
                    if all_val_prompts:
                        columns = ["sample", "clip", "caption"] + [key for key, _ in active_fields]
                        caption_table = wandb.Table(columns=columns)
                        for i, prompt in enumerate(all_val_prompts):
                            if isinstance(prompt, list):  # multi-clip: one row per clip
                                for j, clip_prompt in enumerate(prompt):
                                    row = [i, j, clip_prompt]
                                    for key, scope in active_fields:
                                        val = all_val_meta[key][i]
                                        if scope == "sample":
                                            row.append(val if j == 0 else "")
                                        else:  # per-clip
                                            row.append(str(val[j]))
                                    caption_table.add_data(*row)
                            else:
                                row = [i, 0, prompt]
                                for key, _ in active_fields:
                                    row.append(str(all_val_meta[key][i]))
                                caption_table.add_data(*row)
                        val_wandb_log["validation/captions"] = caption_table
                    wandb.log(val_wandb_log, step=global_update_step)

                barrier()  # wait for main process to finish wandb logging before next training step

                del all_val_outputs, val_outputs, outputs
                torch.cuda.empty_cache()
                gc.collect()

            # Update training step
            global_update_step += 1


if __name__ == "__main__":
    main()
