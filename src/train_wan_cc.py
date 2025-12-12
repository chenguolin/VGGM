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
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import opt_dict, ROOT
from src.data import *  # import all dataset classes and `yield_forever`
from src.models.networks import WanVAEWrapper
from src.models import Wan, DMD_Wan, MyEMAModel, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.vis_util as vis_util


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
        default=f"{ROOT}/projects/VGGM/out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--wandb_token_path",
        type=str,
        default=f"{ROOT}/.cache/wandb/token",
        help="Path to the WandB login token"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
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
        "--max_val_steps",
        type=int,
        default=1,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
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
        "--ema_on_cpu",
        action="store_true",
        help="Load EMA model on CPU for saving memory"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
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
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of the model pretrained in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained model checkpoint"
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

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set args by configs
    args.gradient_accumulation_steps = max(
        args.gradient_accumulation_steps,
        configs["train"].get("gradient_accumulation_steps", 1),
    )
    args.use_deepspeed = (
        args.use_deepspeed or
        configs["train"].get("use_deepspeed", False)
    )
    args.zero_stage = max(
        args.zero_stage,
        configs["train"].get("zero_stage", 2),
    )
    opt.use_deepspeed_zero3 = str(int(args.zero_stage)) == "3"
    args.use_ema = (
        args.use_ema or
        configs["train"].get("use_ema", False)
    )

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = RealcamvidDataset(opt, training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        drop_last=True,
        collate_fn=BaseDataset.collate_fn,
    )
    val_dataset = RealcamvidDataset(opt, training=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,  # shuffle for various visualization
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        drop_last=False,
        collate_fn=BaseDataset.collate_fn,
    )

    logger.info(f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # Initialize the model, optimizer and lr scheduler
    if opt.use_dmd:
        model = DMD_Wan(opt)
    else:
        model = Wan(opt)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(f"Trainable parameter names: {sorted([name for name, param in model.named_parameters() if param.requires_grad])}\n")

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
        optimizer = get_optimizer(params=params_to_optimize, **configs["optimizer"])

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # (Optional) Load a pretrained model
    if args.load_pretrained_model is not None:
        logger.info(f"Load pretrained checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        model, args.load_pretrained_model_ckpt = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_model, "checkpoints"),
            args.load_pretrained_model_ckpt,
            model, accelerator, strict=False,
        )
        # Load a pretrained EMA model
        pretrained_ema_path = os.path.join(args.output_dir, args.load_pretrained_model, "checkpoints", f"{args.load_pretrained_model_ckpt:06d}", "ema_states.pth")
        if os.path.exists(pretrained_ema_path):
            _ema_states = MyEMAModel(
                model.parameters() if not opt.use_dmd else model.diffusion.parameters(),
                use_deepspeed_zero3=str(int(args.zero_stage)) == "3",
                **configs["train"]["ema_kwargs"]
            )
            _ema_states.load_state_dict(torch.load(pretrained_ema_path, map_location="cpu"))
            _ema_states.copy_to(model.parameters() if not opt.use_dmd else model.diffusion.parameters())
            del _ema_states

    # Initialize the EMA model to save moving average states
    if args.use_ema:
        logger.info("Use exponential moving average (EMA) for model parameters\n")
        ema_states = MyEMAModel(
            model.parameters() if not opt.use_dmd else model.diffusion.parameters(),
            use_deepspeed_zero3=str(int(args.zero_stage)) == "3",
            **configs["train"]["ema_kwargs"]
        )
        if not args.ema_on_cpu:
            ema_states.to(accelerator.device)

    # Cast model and dataset to the appropriate dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Prepare VAE
    vae = WanVAEWrapper(opt.vae_path)
    vae = vae.to(device=accelerator.device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()

    # Prepare everything with `accelerator`
    model, optimizer, lr_scheduler, train_loader, val_loader = \
        accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    # assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        if not os.path.exists(os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}')):
            args.resume_from_iter = util.load_ckpt(
                ckpt_dir,
                args.resume_from_iter,
                None,  # `None`: not load model ckpt here
                accelerator,  # manage the process states
            )
        # Load everything
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"), load_kwargs={"weights_only": False})
        global_update_step = int(args.resume_from_iter) + 1
        # Load EMA states
        if args.use_ema and os.path.exists(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "ema_states.pth")):
            ema_states.load_state_dict(torch.load(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}", "ema_states.pth"), map_location="cpu" if args.ema_on_cpu else accelerator.device))

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(model), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
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

    # Start training
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for epoch in range(configs["train"]["epochs"]):

        for batch in train_loader:

            if global_update_step == args.max_train_steps:
                progress_bar.close()
                logger.logger.propagate = True  # propagate to the root logger (console)
                if accelerator.is_main_process:
                    wandb.finish()
                logger.info("Training finished!\n")
                return

            model.train()

            with accelerator.accumulate(model):

                is_eval = ((global_update_step % configs["train"]["early_eval_freq"] == 0 and
                    global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                    or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                    or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                    or global_update_step == args.max_train_steps-1
                )

                if opt.use_dmd:
                    train_generator = global_update_step % opt.generator_train_every == 0
                    outputs = model(batch, dtype=weight_dtype, train_generator=train_generator, is_eval=is_eval, vae=vae)
                else:
                    outputs = model(batch, dtype=weight_dtype, is_eval=is_eval, vae=vae)

                loss = outputs["loss"]

                # Some extra outputs for logging
                critic_loss = outputs["critic_loss"] if "critic_loss" in outputs else None
                generator_loss = outputs["generator_loss"] if "generator_loss" in outputs else None
                dmd_grad_norm = outputs["dmd_grad_norm"] if "dmd_grad_norm" in outputs else None
                diffusion_loss = outputs["diffusion_loss"] if "diffusion_loss" in outputs else None

                # Backpropagate
                accelerator.backward(loss.mean())

                # Gradient clip
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Gather the losses across all processes for logging (if we use distributed training)
                loss = accelerator.gather(loss.detach()).mean()

                if critic_loss is not None:
                    critic_loss = accelerator.gather(critic_loss.detach()).mean()
                if generator_loss is not None:
                    generator_loss = accelerator.gather(generator_loss.detach()).mean()
                if dmd_grad_norm is not None:
                    dmd_grad_norm = accelerator.gather(dmd_grad_norm.detach()).mean()
                if diffusion_loss is not None:
                    diffusion_loss = accelerator.gather(diffusion_loss.detach()).mean()

                logs = {
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.use_ema:
                    if args.ema_on_cpu:
                        ema_states.step([p.cpu() for p in (model.parameters() if not opt.use_dmd else model.diffusion.parameters())])
                    else:
                        ema_states.step(model.parameters() if not opt.use_dmd else model.diffusion.parameters())
                    logs.update({"ema": ema_states.cur_decay_value})

                if critic_loss is not None:
                    logs.update({"critic": critic_loss.item()})
                if generator_loss is not None:
                    logs.update({"generator": generator_loss.item()})
                if diffusion_loss is not None:
                    logs.update({"diffusion": diffusion_loss.item()})

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                logger.info(
                    f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                    f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}" +
                    f", critic: {logs['critic']:.4f}" if critic_loss is not None else "" +
                    f", generator: {logs['generator']:.4f}" if generator_loss is not None else "" +
                    f", diffusion: {logs['diffusion']:.4f}" if diffusion_loss is not None else "" +
                    f", ema: {logs['ema']:.4f}" if args.use_ema else ""
                )

                # Log the training progress
                if (global_update_step % configs["train"]["log_freq"] == 0  # 1. every `log_freq` steps
                    or global_update_step % updated_steps_per_epoch == 0):  # 2. every epoch
                    if accelerator.is_main_process:
                        wandb.log({
                            "training/loss": loss.item(),
                            "training/lr": lr_scheduler.get_last_lr()[0],
                        }, step=global_update_step)
                        if args.use_ema:
                            wandb.log({
                                "training/ema": ema_states.cur_decay_value
                            }, step=global_update_step)
                        if critic_loss is not None:
                            wandb.log({
                                "training/critic_loss": critic_loss.item()
                            }, step=global_update_step)
                        if generator_loss is not None:
                            wandb.log({
                                "training/generator_loss": generator_loss.item()
                            }, step=global_update_step)
                        if dmd_grad_norm is not None:
                            wandb.log({
                                "training/dmd_grad_norm": dmd_grad_norm.item()
                            }, step=global_update_step)
                        if diffusion_loss is not None:
                            wandb.log({
                                "training/diffusion_loss": diffusion_loss.item()
                            }, step=global_update_step)

                # Save checkpoint
                if global_update_step != 0 and (global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                    or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                    or global_update_step == args.max_train_steps-1):  # 3. last step
                    gc.collect()
                    if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                        accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                    elif accelerator.is_main_process:
                        accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                    accelerator.wait_for_everyone()  # ensure all processes have finished saving
                    if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                        if accelerator.is_main_process:
                            if args.use_ema:
                                torch.save(ema_states.state_dict(),
                                    os.path.join(ckpt_dir, f"{global_update_step:06d}", "ema_states.pth"))
                    elif accelerator.is_main_process:
                        if args.use_ema:
                            torch.save(ema_states.state_dict(),
                                os.path.join(ckpt_dir, f"{global_update_step:06d}", "ema_states.pth"))
                    gc.collect()

                # Evaluate on the validation set
                if ((global_update_step % configs["train"]["early_eval_freq"] == 0 and
                    global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                    or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                    or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                    or global_update_step == args.max_train_steps-1):  # 4. last step

                    torch.cuda.empty_cache()
                    gc.collect()

                    # Use EMA parameters for evaluation
                    if args.use_ema:
                        # Store the model parameters temporarily and load the EMA parameters to perform inference
                        ema_states.store(model.parameters() if not opt.use_dmd else model.diffusion.parameters())
                        ema_states.copy_to(model.parameters() if not opt.use_dmd else model.diffusion.parameters())

                    with torch.no_grad():
                        model.eval()

                        all_val_matrics, val_steps = {}, 0
                        val_progress_bar = tqdm(
                            range(len(val_loader)) if args.max_val_steps is None \
                                else range(args.max_val_steps),
                            desc="Validation",
                            ncols=125,
                            disable=not accelerator.is_main_process
                        )
                        for val_batch in val_loader:
                            val_outputs = model(val_batch, func_name="evaluate", vae=vae)

                            val_logs = {}
                            if "image" in val_batch:
                                for cfg_scale in opt.cfg_scale:
                                    val_psnr = val_outputs[f"psnr_{cfg_scale}"]
                                    val_ssim = val_outputs[f"ssim_{cfg_scale}"]

                                    val_psnr = accelerator.gather_for_metrics(val_psnr).mean()
                                    val_ssim = accelerator.gather_for_metrics(val_ssim).mean()

                                    val_logs.update({
                                        f"psnr_{cfg_scale}": val_psnr.item(),
                                        f"ssim_{cfg_scale}": val_ssim.item()
                                    })
                                    all_val_matrics.setdefault(f"psnr_{cfg_scale}", []).append(val_psnr)
                                    all_val_matrics.setdefault(f"ssim_{cfg_scale}", []).append(val_ssim)

                            val_progress_bar.set_postfix(**val_logs)
                            val_progress_bar.update(1)
                            val_steps += 1

                            if args.max_val_steps is not None and val_steps == args.max_val_steps:
                                break

                    val_progress_bar.close()

                    if args.use_ema:
                        # Switch back to the original model parameters
                        ema_states.restore(model.parameters() if not opt.use_dmd else model.diffusion.parameters())

                    for k, v in all_val_matrics.items():
                        all_val_matrics[k] = torch.tensor(v).mean()

                    if "image" in val_batch:
                        for cfg_scale in opt.cfg_scale:
                            logger.info(
                                f"Eval [{global_update_step:06d} / {total_updated_steps:06d}] " +
                                f"psnr_{cfg_scale}: {all_val_matrics[f'psnr_{cfg_scale}'].item():.4f}, " +
                                f"ssim_{cfg_scale}: {all_val_matrics[f'ssim_{cfg_scale}'].item():.4f}\n"
                            )

                    outputs = accelerator.gather(outputs)
                    val_outputs = accelerator.gather_for_metrics(val_outputs)

                    if accelerator.is_main_process:
                        if "image" in val_batch:
                            for cfg_scale in opt.cfg_scale:
                                wandb.log({
                                    f"validation/psnr_{cfg_scale}": all_val_matrics[f"psnr_{cfg_scale}"].item(),
                                    f"validation/ssim_{cfg_scale}": all_val_matrics[f"ssim_{cfg_scale}"].item()
                                }, step=global_update_step)

                        # Visualization
                        wandb.log({
                            "videos/training": vis_util.wandb_video_log(outputs, fps=8)
                        }, step=global_update_step)
                        wandb.log({
                            "videos/validation": vis_util.wandb_video_log(val_outputs, fps=8)
                        }, step=global_update_step)

                    torch.cuda.empty_cache()
                    gc.collect()

                # Update training step
                global_update_step += 1


if __name__ == "__main__":
    main()
