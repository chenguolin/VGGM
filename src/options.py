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
    version_2sdiff: bool = False
    version_action: bool = False
    version_new_action: bool = False
    action_data_path: str = "video_action_caption_70w_p1.jsonl"  # relative to dataroot
    load_global_caption: bool = False
    num_clips: int = 1
    num_clips_test: Optional[int] = None
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
    da3_interactive: bool = True
    da3_input_cam: bool = False
        ## Train
    da3_weight_type: Literal[
        "uniform",
        "diffusion",
        "inverse_timestep",
    ] = "uniform"
    da3_down_ratio: int = 1
    da3_chunk_size: int = 8  # DPT head chunk size, not for causality
    da3_use_ray_pose: bool = False
        ## Self Geometry Forcing
    da3_loss_in_sf: bool = False
    render_loss_in_sf: bool = False

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
    input_timestamps: bool = False
    input_pcrender: bool = False
    first_latent_cond: bool = False
    random_i2v_prob: float = 1.
    conf_thresh_percentile: float = 0.4
    rand_pcrender_ratio: float = 1.
    min_num_points: int = 1000
    max_num_points: int = 100000
        ## Causal
    is_causal: bool = False
    sink_size: int = 0
    chunk_size: int = 3
    max_window_size: int = None  # if None, then `(num_input_frames - 1) // 4 + 1`
    max_kvcache_size: Optional[int] = None  # if None, then `max_window_size + sink_size`
    rope_outside: bool = False
    use_flexattn: bool = True  # set to False to save memory; `block_mask` in flex_attn takes so much memory!
    prefill_image: bool = True
        ## TTT
    use_ttt: bool = False
    ttt_layers: Optional[str] = None  # e.g. "0,2,4,6,8"; None means all layers when `use_ttt=True`
    ttt_num_fw_heads: int = 8
    ttt_fw_head_dim: Optional[int] = None  # None = same as attention `head_dim`
    ttt_chunk_size: Optional[int] = None  # None = `frame_seqlen * chunk_size`
    ttt_w0_w2_low_rank: int = 32
    ttt_use_muon: bool = True
    ttt_use_momentum: bool = True
    ttt_prenorm: bool = True
    ttt_use_conv: bool = False
    ttt_conv_kernel: int = 3
        ## GatedDeltaNet
    use_gdn: bool = False
    gdn_layers: Optional[str] = None  # e.g. "0,2,4,6,8"; None means all layers when `use_gdn=True`
    gdn_num_heads: int = 8
    gdn_head_qk_dim: Optional[int] = None  # None = same as attention `head_dim`
    gdn_head_v_dim: Optional[int] = None   # None = same as `gdn_head_qk_dim`
    gdn_causal_mode: Literal["bidirectional", "causal"] = "bidirectional"
    gdn_chunk_size: Optional[int] = None  # None = `frame_seqlen * chunk_size`
    gdn_use_conv: bool = False
    gdn_conv_kernel: int = 3
        ## Attention gate (for progressive GDN/TTT-only transition)
    attn_gate_layers: Optional[str] = None  # e.g. "0,2,4,6,8"; layers where attention output gets a learnable gate (init 1.)
        ## Load pre-trained models
    generator_path: Optional[str] = None
    lora_path: Optional[str] = None
    teacher_path: Optional[str] = None
    fake_path: Optional[str] = None
    fake_lora_path: Optional[str] = None
    is_teacher_causal: bool = False
    teacher_use_teacher_forcing: bool = False
    teacher_input_plucker: bool = False
    teacher_input_timestamps: bool = False
    teacher_input_pcrender: bool = False
    teacher_first_latent_cond: bool = False
        ## DMD
    use_dmd: bool = False
    generator_train_every: int = 5
    fake_guidance_scale: float = 1.
    real_guidance_scale: float = 4.
    separate_gen_crit: bool = False  # to reduce peak memory
    real_score_offload: bool = False  # to save memory
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
    eval_offline: bool = False
    cfg_dropout: float = 0.
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
        ## Freeze the first N layers of `self.diffusion.model`; only the last layers remain trainable
    num_trainable_last_layers: Optional[int] = None
        ## LoRA for DMD fake score model
    use_lora_in_fake_score: bool = False
    lora_target_modules_in_fake_score: str = "q,k,v,o,ffn.0,ffn.2"
    lora_rank_in_fake_score: int = 32
        ## Trainable modules except LoRA layers for DMD fake score model
    more_trainable_fake_score_params: Optional[str] = None
        ## Freeze the first N layers of `self.fake_score.model`; only the last layers remain trainable
    num_trainable_last_fake_score_layers: Optional[int] = None

    # Training
        ## Sequence parallel
    sp_size: int = 1
        ## Critic loss in DMD
    critic_loss_weight: float = 1.
        ## Diffusion loss in DMD
    diffusion_loss_prob: float = 0.
    diffusion_loss_weight: float = 1.
        ## DA3 losses
    conf_alpha: float = 0.2
    gradient_loss_scale: int = 4
    xyz_loss_threshold: float = 10.
    depth_loss_threshold: float = 10.
    camera_loss_threshold: float = 10.
        ## LR scheduler
    name_lr_mult: Optional[str] = None
    exclude_name_lr_mult: Optional[str] = None
    lr_mult: float = 0.1

    # Misc
    git_version: str = None  # will fill by `util.get_git_version()` in the main process

    def __post_init__(self):
        if self.num_input_frames_test is None:
            self.num_input_frames_test = self.num_input_frames

        if self.num_clips_test is None:
            self.num_clips_test = self.num_clips

        if self.load_da3:
            self.load_depth = True

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

        # TTT
        if self.use_ttt:
            if self.ttt_chunk_size is None:
                self.ttt_chunk_size = self.frame_seqlen * self.chunk_size
            # Parse `ttt_layers` from comma-separated string to list of ints
            if self.ttt_layers is not None:
                self.ttt_layers_list = [int(x) for x in self.ttt_layers.split(",")]
            else:
                self.ttt_layers_list = None  # will be resolved after model is loaded
        else:
            self.ttt_layers_list = None

        # GatedDeltaNet
        if self.use_gdn:
            if self.gdn_chunk_size is None:
                self.gdn_chunk_size = self.frame_seqlen * self.chunk_size
            if self.gdn_layers is not None:
                self.gdn_layers_list = [int(x) for x in self.gdn_layers.split(",")]
            else:
                self.gdn_layers_list = None  # will be resolved after model is loaded
        else:
            self.gdn_layers_list = None

        # Attention gate
        if self.attn_gate_layers is not None:
            self.attn_gate_layers_list = [int(x) for x in self.attn_gate_layers.split(",")]
        else:
            self.attn_gate_layers_list = None


# Set all options for different tasks and models
opt_dict: Dict[str, Options] = {}


# Wan2.1-T2V-1.3B
opt_dict["wan2.1_t2v"] = Options(
    eval_offline=False,  # True
    #
    use_internal_dataset=True,
    version_new_action=True,
    #
    num_input_frames=81,
    # input_res=(480, 832),
    #
    num_clips=1,
    sp_size=1,
    shift=5.,
    #
    wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    # generator_path=f"{ROOT}/projects/VGGM/.pth",
    #
    is_causal=False,  # True
    use_teacher_forcing=False,  # True
    #
    sink_size=3,
    chunk_size=3,
    max_window_size=9,
    rope_outside=True,
    use_flexattn=True,  # False
    #
    load_da3=False,  # True
    da3_interactive=True,  # False
    da3_weight_type="uniform",
    da3_down_ratio=1,
    # only_train_da3=True,
    # no_noise_for_da3=True,
    only_train_resdit=False,  # True
    #
    use_ttt=False,  # True
    #
    use_gdn=False,  # True
    #
    input_plucker=False,  # True
    input_timestamps=True,  # False
    exclude_name_lr_mult="gdn_branch,ttt_branch,plucker_embed,timestamp_embed,extra_condition_embed",
    #
    # load_conf=True,
    # input_pcrender=True,
    # load_tae=True,
)

# Self-Forcing DMD
opt_dict["wan2.1_t2v_dmd"] = Options(
    eval_offline=False,  # True
    #
    use_internal_dataset=True,
    version_new_action=True,
    #
    num_input_frames=81,
    # input_res=(480, 832),
    #
    num_clips=1,
    sp_size=8,
    shift=5.,
    #
    only_static_data=False,
    use_vidprom=False,  # True
    vidprom_prob=1.,  # 0.5
    use_short_caption=False,
    first_latent_cond=False,
    input_plucker=False,  # True
    input_timestamps=True,  # False
    #
    load_da3=False,  # True
    da3_interactive=True,  # False
    da3_weight_type="uniform",
    da3_down_ratio=1,
    only_train_resdit=False,  # True
    #
    da3_loss_in_sf=False,  # True
    render_loss_in_sf=False,  # True
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
    use_flexattn=True,  # False
    #
    wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    real_wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    fake_wan_dir=f"{ROOT}/.cache/huggingface/hub/Wan-AI/Wan2.1-T2V-14B",
    #
    generator_path=f"{ROOT}/projects/VGGM/.pth",
    teacher_path=f"{ROOT}/projects/VGGM/.pth",
    fake_path=f"{ROOT}/projects/VGGM/.pth",
    #
    is_teacher_causal=False,
    teacher_use_teacher_forcing=False,
    teacher_input_plucker=True,
    teacher_first_latent_cond=False,
    #
    use_dmd=True,
    separate_gen_crit=False,  # True
    real_score_offload=False,  # True
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
    use_lora_in_wan=False,  # True
    lora_target_modules_in_wan="q,k,v,o,ffn.0,ffn.2",
    lora_rank_in_wan=32,
    num_trainable_last_layers=None,  # 10
    #
    use_lora_in_fake_score=False,  # True
    lora_target_modules_in_fake_score="q,k,v,o,ffn.0,ffn.2",
    lora_rank_in_fake_score=32,
    num_trainable_last_fake_score_layers=None,  # 10
    #
    # load_conf=True,
    # input_pcrender=True,
    # load_tae=True,
    #
    critic_loss_weight=1.,
    #
    diffusion_loss_prob=0.,
    diffusion_loss_weight=1.,
)
