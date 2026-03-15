from typing import *

import os
from dataclasses import dataclass

ROOT_LIST = [
    "/apdcephfs_sgfd/share_303967936/cglin",
    "/apdcephfs/share_sg/apdcephfs_sgfd/share_303967936/cglin",
]

ROOT = None
for root_candidate in ROOT_LIST:
    if os.path.exists(root_candidate):
        ROOT = root_candidate
        break
if ROOT is None:
    raise ValueError(f"None of the following roots exist: {ROOT_LIST}")

DATAROOT_LIST = [
    "/apdcephfs_sgfd/share_303967936/世界模型实验数据",
    "/apdcephfs/share_sg/apdcephfs_sgfd/share_303967936/世界模型实验数据",
]

DATAROOT = None
for root_candidate in DATAROOT_LIST:
    if os.path.exists(root_candidate):
        DATAROOT = root_candidate
        break
if DATAROOT is None:
    raise ValueError(f"None of the following data roots exist: {DATAROOT_LIST}")


@dataclass
class Options:
    # Data
    input_res: Tuple[int, int] = (288, 512)
    size_divisor: int = 16  # required for Wan2.1
    num_input_frames: int = 81
    num_input_frames_test: Optional[int] = None
    crop_resize_ratio: Tuple[float, float] = (0.77, 1.)
    pingpong_threshold: int = 64
    load_image: bool = True
        ## Interal
    use_internal_dataset: bool = False
    version_2s35w: bool = False
    num_clips: int = 1
    random_num_clips: bool = False
        ## RealCamVid
    load_da3_cam: bool = True
    load_depth: bool = True
    load_conf: bool = False
    normalize_xyz: bool = True
    use_short_caption: bool = False
    only_static_data: bool = False
        ## VidProm
    use_vidprom: bool = False
    vidprom_prob: float = 0.5
        ## Camera normalization
    camera_norm_type: Literal[
        "none",
        "canonical",
    ] = "canonical"
    camera_norm_unit: float = 1.
        ## Path
    root: str = f"{ROOT}/data"
    dataroot: str = DATAROOT
        ## Post initialization (`__post_init__`)
    dataset_dir_train: str = None
    dataset_dir_test: str = None

    # DA3
    load_da3: bool = False
        ### Model
    da3_model_name: str = "da3-large-1.1"
    fix_da3_heads: bool = True
    fix_shared_dit_layers: bool = False
    only_train_da3: bool = False
    only_train_resdit: bool = False
    no_noise_for_da3: bool = False
    da3_interactive: bool = False
    da3_input_cam: bool = True
        ## Train
    da3_weight_type: Literal[
        "uniform",
        "diffusion",
        "inverse_timestep",
    ] = "inverse_timestep"
    da3_down_ratio: int = 1
    da3_chunk_size: int = 8  # DPT head chunk size, not for causality
    da3_use_ray_pose: bool = False
        ## Self Geometry Forcing
    da3_loss_in_sf: bool = False
    render_loss_in_sf: bool = True

    # VAE
    vae_path: str = f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
        ## TAE
    tae_path: str = f"{ROOT}/projects/VGGM/resources/taew2_1.pth"
    load_tae: bool = False
        ## Post initialization (`__post_init__`)
    compression_ratio: Tuple[int, int, int] = None
    latent_dim: int = None

    # Wan
    wan_dir: str = f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B"
    real_wan_dir: str = f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B"
    fake_wan_dir: str = f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B"
        ## Model
    load_text_encoder: bool = True
    input_plucker: bool = False
    input_pcrender: bool = False
    first_latent_cond: bool = False
    random_i2v_prob: float = 1.
    enable_riflex: bool = False
    conf_thresh_percentile: float = 0.4
    rand_pcrender_ratio: float = 1.
    min_num_points: int = 1000
    max_num_points: int = 100000
        ## DDT
    use_ddt: bool = False
    ddt_num_layers: int | float = 0.1  # int: number of layers; float: ratio in [0,1]
    ddt_fusion: bool = True
        ## Causal
    is_causal: bool = False
    sink_size: int = 0
    chunk_size: int = 3
    max_window_size: int = None  # if None, then `(num_input_frames - 1) // 4 + 1`
    max_kvcache_size: int = 21  # set to a limited number to save memory
    rope_outside: bool = False
    prefill_image: bool = True
        ## Load pre-trained models
    generator_path: Optional[str] = None
    lora_path: Optional[str] = None  # Path to LoRA weights file (lora_weights.pth)
    teacher_path: Optional[str] = None
    fake_path: Optional[str] = None
    fake_lora_path: Optional[str] = None  # Path to fake_score LoRA weights file
    is_teacher_causal: bool = False
    teacher_input_plucker: bool = False
    teacher_input_pcrender: bool = False
    teacher_first_latent_cond: bool = False
        ## DMD
    use_dmd: bool = False
    generator_train_every: int = 5
    fake_guidance_scale: float = 1.
    real_guidance_scale: float = 4.
    ddt_fake_score: bool = False
        ## Self-forcing
    self_forcing_prob: float = 1.
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
    cfg_scale: Tuple[float, ...] = (5.,)
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    deterministic_inference: bool = True
    use_lpips: bool = False  # False to save memory
        ## LoRA
    save_lora_only: bool = True
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
        ## Sequence parallel
    sp_size: int = 1
        ## Critic loss in DMD
    critic_loss_weight: float = 1.
        ## Diffusion loss in DMD
    diffusion_loss_prob: float = 0.
    diffusion_loss_weight: float = 1.
    ddt_diffusion_loss: bool = False
        ## DA3 losses
    conf_alpha: float = 0.2
    gradient_loss_scale: int = 4
    xyz_loss_threshold: float = 10.
    depth_loss_threshold: float = 10.
    camera_loss_threshold: float = 10.
        ## Self-supervised loss
    self_supervised_loss_weight: float = 0.
    student_layer_idx: Optional[int | float] = None  # int: block index; float: ratio in [0,1]
    teacher_layer_idx: Optional[int | float] = None  # int: block index; float: ratio in [0,1]
    self_supervised_mask_ratio: float = 0.1
        ## LR scheduler
    name_lr_mult: Optional[str] = None
    exclude_name_lr_mult: Optional[str] = None
    lr_mult: float = 0.1

    # Misc
    git_version: str = None  # will fill by `util.get_git_version()` in the main process

    def __post_init__(self):
        if self.num_input_frames_test is None:
            self.num_input_frames_test = self.num_input_frames

        if self.load_da3:
            self.load_depth = True

        # DDT multi-head
        if self.ddt_diffusion_loss or self.ddt_fake_score:
            self.use_ddt = True

        # Dataset directories
        self.dataset_dir_train = {
            "realcamvid": f"{self.root}/RealCam-Vid",
            "internal": self.dataroot,
        }
        self.dataset_dir_test = {
            "realcamvid": f"{self.root}/RealCam-Vid",
            "internal": self.dataroot,
        }

        # Extra condition
        self.extra_condition_dim = 5 if self.input_pcrender else 0
        self.teacher_extra_condition_dim = 5 if self.teacher_input_pcrender else 0

        # VAE
        if "Wan2.2_VAE.pth" in self.vae_path:
            self.compression_ratio = (4, 16, 16)
            self.latent_dim = 48
        elif "Wan2.1_VAE.pth" in self.vae_path:
            self.compression_ratio = (4, 8, 8)
            self.latent_dim = 16

        # Attention
        if self.max_window_size is None:
            self.max_window_size = (self.num_input_frames - 1) // self.compression_ratio[0] + 1 - self.sink_size
        if self.max_kvcache_size is None:
            self.max_kvcache_size = self.max_window_size + self.sink_size
        self.frame_seqlen = (
            self.input_res[0] // self.compression_ratio[1] // 2 *  # `2`: patch size in DiT is hard-coded to 2
            self.input_res[1] // self.compression_ratio[2] // 2
        )
        self.max_attention_size = (self.max_window_size + self.sink_size) * self.frame_seqlen
        self.da3_max_attention_size = (self.max_window_size + self.sink_size) * (self.frame_seqlen // (self.da3_down_ratio * self.da3_down_ratio) + 1)  # `+1` for camera token
        self.max_kvcache_attention_size = self.max_kvcache_size * self.frame_seqlen
        self.da3_max_kvcache_attention_size = self.max_kvcache_size * (self.frame_seqlen // (self.da3_down_ratio * self.da3_down_ratio) + 1)  # `+1` for camera token


# Set all options for different tasks and models
opt_dict: Dict[str, Options] = {}


# Wan2.1-T2V-1.3B
opt_dict["wan2.1_t2v_1.3b"] = Options(
    # use_internal_dataset=True,
    #
    # input_res=(480, 832),
    #
    num_clips=1,
    sp_size=1,
    #
    # wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    # generator_path=f"{ROOT}/projects/VGGM/.pth",
    #
    input_plucker=True,
    exclude_name_lr_mult="plucker_embed,extra_condition_embed",
    #
    # load_conf=True,
    # input_pcrender=True,
)

# Diffusion/Teacher Forcing
opt_dict["wan2.1_t2v_1.3b_causal"] = Options(
    # use_internal_dataset=True,
    #
    # input_res=(480, 832),
    #
    num_clips=1,
    sp_size=1,
    #
    load_da3=False,  # True
    da3_interactive=False,
    da3_weight_type="inverse_timestep",
    da3_down_ratio=1,
    # only_train_da3=True,
    # no_noise_for_da3=True,
    #
    is_causal=True,
    use_teacher_forcing=True,  # False
    #
    sink_size=3,
    chunk_size=3,
    max_window_size=9,
    rope_outside=True,
    #
    # wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    generator_path=f"{ROOT}/projects/VGGM/.pth",
    #
    input_plucker=True,
    exclude_name_lr_mult="plucker_embed,extra_condition_embed,da3_adapter",
    #
    # num_input_frames=201,
    # num_input_frames_test=201,
    #
    # load_conf=True,
    # input_pcrender=True,
    # load_tae=True,
)

# Self-Forcing DMD
opt_dict["wan2.1_t2v_1.3b_dmd"] = Options(
    # use_internal_dataset=True,
    #
    # input_res=(480, 832),
    #
    num_clips=1,
    max_kvcache_size=21,
    sp_size=1,
    #
    only_static_data=False,
    use_vidprom=False,  # True
    vidprom_prob=1.,  # 0.5
    use_short_caption=False,
    first_latent_cond=False,
    input_plucker=True,
    #
    load_da3=False,  # True
    da3_interactive=False,
    da3_weight_type="uniform",
    da3_down_ratio=1,
    only_train_resdit=True,
    #
    da3_loss_in_sf=False,  # True
    render_loss_in_sf=False,
    #
    # diffusion_loss_prob=0.,
    # no_noise_for_da3=False,
    #
    is_causal=True,
    use_teacher_forcing=False,
    #
    sink_size=3,
    chunk_size=3,
    max_window_size=9,
    rope_outside=True,
    #
    # wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    # real_wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    # fake_wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    #
    generator_path=f"{ROOT}/projects/VGGM/.pth",
    teacher_path=f"{ROOT}/projects/VGGM/.pth",
    fake_path=f"{ROOT}/projects/VGGM/.pth",
    #
    is_teacher_causal=False,
    teacher_input_plucker=True,
    teacher_first_latent_cond=False,
    #
    use_dmd=True,
    self_forcing_prob=1.,
    real_guidance_scale=4.,
    last_step_only=False,
    context_noise=0,
    same_step_across_chunks=True,
    #
    name_lr_mult="fake_score",
    lr_mult=0.2,
    #
    # num_input_frames_test=201,
    num_inference_steps=4,
    cfg_scale=(1.,),
    deterministic_inference=False,
    #
    # load_conf=True,
    # input_pcrender=True,
    # load_tae=True,
    #
    critic_loss_weight=1.,
    ddt_fake_score=False,
    ddt_num_layers=0.1,
    ddt_fusion=True,
    #
    diffusion_loss_prob=0.,
    diffusion_loss_weight=1.,
    ddt_diffusion_loss=False,
)

# Self-Forcing reproduction
opt_dict["sf_rep"] = Options(
    input_res=(480, 832),
    #
    is_causal=True,
    use_teacher_forcing=False,
    #
    sink_size=0,
    chunk_size=3,
    max_window_size=21,
    rope_outside=False,
    #
    generator_path=f"{ROOT}/.cache/ode_init.pt",
    real_wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    teacher_path=None,
    is_teacher_causal=False,
    #
    use_dmd=True,
    self_forcing_prob=1.,
    real_guidance_scale=4.,
    last_step_only=False,
    context_noise=0,
    same_step_across_chunks=True,
    #
    name_lr_mult="fake_score",
    lr_mult=0.2,
    #
    num_input_frames_test=81,
    num_inference_steps=4,
    cfg_scale=(1.,),
    deterministic_inference=False,
)
