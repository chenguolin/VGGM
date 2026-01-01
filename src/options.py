from typing import *

import os
from dataclasses import dataclass

ROOT_LIST = [
    "/apdcephfs_fsgm/share_303967936/cglin",
    "/apdcephfs_sgfd/share_303967936/cglin",
    "/apdcephfs/share_gz/apdcephfs_fsgm/share_303967936/cglin",
    "/apdcephfs/share_sg/apdcephfs_sgfd/share_303967936/cglin",
]

ROOT = None
for root_candidate in ROOT_LIST:
    if os.path.exists(root_candidate):
        ROOT = root_candidate
        break
if ROOT is None:
    raise ValueError(f"None of the following roots exist: {ROOT_LIST}")


@dataclass
class Options:
    # Data
    input_res: Tuple[int, int] = (288, 512)
    size_divisor: int = 16  # required for Wan2.1
    num_input_frames: int = 81
    num_input_frames_test: Optional[int] = None
    crop_resize_ratio: Tuple[float, float] = (0.77, 1.)
    load_da3_cam: bool = True
    load_depth: bool = True
    normalize_xyz: bool = True
    use_vidprom: bool = False
        ## Camera normalization
    camera_norm_type: Literal[
        "none",
        "canonical",
    ] = "canonical"
    camera_norm_unit: float = 1.
    root: str = f"{ROOT}/data"
    ode_pairs_dir: Optional[str] = None
        ## Post initialization (`__post_init__`)
    dataset_dir_train: str = None
    dataset_dir_test: str = None

    # DA3
    da3_model_name: str = "da3-large-1.1"
    da3_chunk_size: int = 8  # DPT head chunk size, not for causality
    da3_use_ray_pose: bool = False
    da3_use_bicrossattn: bool = True
    load_da3: bool = False
    only_train_da3: bool = False

    # VAE
    vae_path: str = f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        ## Post initialization (`__post_init__`)
    compression_ratio: Tuple[int, int, int] = None
    latent_dim: int = None

    # Wan
    wan_dir: str = f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B"
        ## Model
    load_text_encoder: bool = True
    load_clip_encoder: bool = True
    input_plucker: bool = False
    first_latent_cond: bool = False
    random_i2v_prob: float = 1.
    enable_riflex: bool = False
        ## Memory
    memory_num_tokens: int = 0
    memory_num_blocks: int = 6
    memory_dim: int = 1024
    memory_num_heads: int = 8
        ## Causal
    is_causal: bool = False
    sink_size: int = 0
    chunk_size: int = 3
    max_window_size: int = None  # if None, then `(num_input_frames - 1) // 4 + 1`
    max_kvcache_size: int = 21  # set to a limited number to save memory
    rope_outside: bool = False
        ## Load pre-trained models
    generator_path: Optional[str] = None
    teacher_path: Optional[str] = None
    is_teacher_causal: bool = False
    teacher_input_plucker: bool = False
    teacher_first_latent_cond: bool = False
        ## DMD
    use_dmd: bool = False
    generator_train_every: int = 5
    fake_guidance_scale: float = 1.
    real_guidance_scale: float = 4.
    dmd_loss_weight: float = 1.
        ## Self-forcing
    self_forcing: bool = True
    denoising_step_list: Tuple[int, ...] = (1000, 750, 500, 250)
    warp_denoising_step: bool = True
    last_step_only: bool = False
    context_noise: int = 0
    same_step_across_chunks: bool = True
        ## Teacher-forcing
    use_teacher_forcing: bool = False
        ## Noise scheduler
    num_train_timesteps: int = 1000
    num_inference_steps: int = 25
    min_timestep_boundary: float = 0.
    max_timestep_boundary: float = 1.
    shift: float = 5.
        ## Gradient checkpointing
    use_gradient_checkpointing: bool = True
    use_gradient_checkpointing_offload: bool = False
        ## Training and inference
    cfg_dropout: float = 0.1
    cfg_scale: Tuple[float, ...] = (1., 3., 5.,)
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    deterministic_inference: bool = True
    use_lpips: bool = False  # False to save memory
        ## LoRA
    use_lora_in_wan: bool = False
    lora_target_modules_in_wan: str = "q,k,v,o,ffn.0,ffn.2"
    lora_rank_in_wan: int = 32
        ## Trainable modules except LoRA layers
    more_trainable_wan_params: Optional[str] = None
        ## LoRA for DMD fake score model
    use_lora_in_fake_score: bool = False
    lora_target_modules_in_fake_score: str = "q,k,v,o,ffn.0,ffn.2"
    lora_rank_in_fake_score: int = 32
        ## Trainable modules except LoRA layers for DMD fake score model
    more_trainable_fake_score_params: Optional[str] = None

    # Training
        ## Losses
    conf_alpha: float = 0.2
    gradient_loss_scale: int = 4
    xyz_loss_threshold: float = 1.
    depth_loss_threshold: float = 1.
    camera_loss_threshold: float = 1.
        ## LR scheduler
    name_lr_mult: Optional[str] = None
    exclude_name_lr_mult: Optional[str] = None
    lr_mult: float = 0.1
        ## DeepSpeed
    use_deepspeed_zero3: bool = False

    # Misc
    git_version: str = None  # will fill by `util.get_git_version()` in the main process

    def __post_init__(self):
        if self.num_input_frames_test is None:
            self.num_input_frames_test = self.num_input_frames

        if self.load_da3:
            self.load_depth = True

        # Dataset directories
        self.dataset_dir_train = {
            "realcamvid": f"{self.root}/RealCam-Vid",
        }
        self.dataset_dir_test = {
            "realcamvid": f"{self.root}/RealCam-Vid",
        }

        # VAE
        if "Wan2.2_VAE.pth" in self.vae_path:
            self.compression_ratio = (4, 16, 16)
            self.latent_dim = 48
        elif "Wan2.1_VAE.pth" in self.vae_path:
            self.compression_ratio = (4, 8, 8)
            self.latent_dim = 16

        if self.max_window_size is None:
            self.max_window_size = (self.num_input_frames - 1) // self.compression_ratio[0] + 1 - self.sink_size
        if self.max_kvcache_size is None:
            self.max_kvcache_size = self.max_window_size + self.sink_size
        self.frame_seqlen = (
            self.input_res[0] // self.compression_ratio[1] // 2 *  # `2`: patch size in DiT is hard-coded to 2
            self.input_res[1] // self.compression_ratio[2] // 2
        )

        self.max_attention_size = (self.max_window_size + self.sink_size) * self.frame_seqlen
        self.da3_max_attention_size = (self.max_window_size + self.sink_size) * (self.frame_seqlen + 1)  # `+1` for camera token
        self.max_kvcache_attention_size = self.max_kvcache_size * self.frame_seqlen
        self.da3_max_kvcache_attention_size = self.max_kvcache_size * (self.frame_seqlen + 1)  # `+1` for camera token


# Set all options for different tasks and models
opt_dict: Dict[str, Options] = {}


# Wan2.1-T2V-1.3B
opt_dict["wan2.1_t2v_1.3b"] = Options(
    input_plucker=True,
    name_lr_mult="diffusion.model",
)
opt_dict["wan2.1_t2v_1.3b_i2v"] = Options(
    first_latent_cond=True,
    input_plucker=True,
    name_lr_mult="diffusion.model",
)
opt_dict["wan2.1_t2v_1.3b_i2v_causal"] = Options(
    first_latent_cond=True,
    input_plucker=True,
    #
    is_causal=True,
    use_teacher_forcing=True,
    #
    generator_path=f"{ROOT}/projects/VGGM/.pth",
)

# Self-Forcing reproduction
opt_dict["sf_rep"] = Options(
    is_causal=True,
    generator_path=f"{ROOT}/.cache/ode_init.pt",
    use_dmd=True,
    num_inference_steps=4,
    deterministic_inference=False,
    name_lr_mult="fake_score",
    lr_mult=0.2,
)

# ODE Distillation
opt_dict["wan2.1_t2v_1.3b_ode"] = Options(
    input_plucker=True,
    #
    is_causal=True,
    use_teacher_forcing=False,
    #
    sink_size=0,
    chunk_size=3,
    max_window_size=21,
    rope_outside=False,
    #
    generator_path=f"{ROOT}/projects/VGGM/.pth",
    ode_pairs_dir=f"{ROOT}/data/ode_pairs_t2v",
    #
    num_inference_steps=4,
    deterministic_inference=True,
)

# Self-Forcing DMD
opt_dict["wan2.1_t2v_1.3b_dmd"] = Options(
    use_vidprom=False,
    first_latent_cond=False,
    input_plucker=True,
    #
    is_causal=True,
    use_teacher_forcing=False,
    #
    sink_size=0,
    chunk_size=3,
    max_window_size=21,
    rope_outside=False,
    #
    generator_path=f"{ROOT}/projects/VGGM/.pth",
    teacher_path=f"{ROOT}/projects/VGGM/.pth",
    is_teacher_causal=False,
    teacher_input_plucker=True,
    teacher_first_latent_cond=False,
    #
    use_dmd=True,
    self_forcing=True,
    real_guidance_scale=4.,
    last_step_only=False,
    context_noise=0,
    same_step_across_chunks=True,
    #
    name_lr_mult="fake_score",
    lr_mult=0.2,
    #
    num_inference_steps=4,
    deterministic_inference=False,
)
