#!/bin/bash
# Fine-tune Qwen3-VL for structured caption prediction (multi-task).
#
# Multi-task: each sample has its own system prompt in messages[0],
# so we do NOT pass a global --system flag to ms-swift.
#
# Tasks trained jointly:
#   1. Next-segment prediction (caption + duration estimation)
#   2. Action completion judge
#
# Usage:
#   bash resources/qwen_vl/finetune.sh <data_jsonl> <output_dir> [options]
#
# Options:
#   --model_size 2B|4B|8B  Model size (default: 2B)
#   --tuner lora|full      Training mode (default: lora)
#   --freeze_vit           Freeze ViT (default: true for lora, false for full)
#   --no_freeze_vit        Unfreeze ViT
#   --freeze_aligner       Freeze aligner (default: true for lora, false for full)
#   --no_freeze_aligner    Unfreeze aligner
#   --num_gpus N           Number of GPUs (default: 1)
#   --epochs N             Number of training epochs (default: 3)
#   --grad_accum N         Gradient accumulation steps (default: 16)
#   --save_steps N         Save checkpoint every N steps (default: 500)
#   --lr RATE              Override learning rate
#   --lora_rank N          LoRA rank (default: 16)
#
# Examples:
#   # Prepare data first
#   python resources/qwen_vl/prepare_sft_data.py \
#       --output_dir /tmp/qwen_vl_sft --training_split --num_workers 8
#
#   # LoRA on 2B (default, ~8GB VRAM)
#   bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_lora
#
#   # LoRA on 8B, 2 GPUs
#   bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_8b \
#       --model_size 8B --num_gpus 2
#
#   # Full fine-tuning on 2B, freeze ViT + aligner (only train LLM)
#   bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_full \
#       --tuner full --freeze_vit --freeze_aligner

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────
if [ $# -lt 2 ]; then
    echo "Usage: finetune.sh <data_jsonl> <output_dir> [options]"
    exit 1
fi

DATA_JSONL="$1"; shift
OUTPUT_DIR="$1"; shift

TUNER_TYPE="lora"
MODEL_SIZE="2B"
FREEZE_VIT=""
FREEZE_ALN=""
NUM_GPUS=1
NUM_EPOCHS=3
GRAD_ACCUM=16
SAVE_STEPS=500
LR_OVERRIDE=""
LORA_RANK=16
MAX_STEPS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --tuner)         TUNER_TYPE="$2"; shift 2 ;;
        --model_size)    MODEL_SIZE="$2"; shift 2 ;;
        --freeze_vit)    FREEZE_VIT="true"; shift ;;
        --no_freeze_vit) FREEZE_VIT="false"; shift ;;
        --freeze_aligner)    FREEZE_ALN="true"; shift ;;
        --no_freeze_aligner) FREEZE_ALN="false"; shift ;;
        --num_gpus)      NUM_GPUS="$2"; shift 2 ;;
        --epochs)        NUM_EPOCHS="$2"; shift 2 ;;
        --grad_accum)    GRAD_ACCUM="$2"; shift 2 ;;
        --save_steps)    SAVE_STEPS="$2"; shift 2 ;;
        --lr)            LR_OVERRIDE="$2"; shift 2 ;;
        --lora_rank)     LORA_RANK="$2"; shift 2 ;;
        --max_steps)     MAX_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$TUNER_TYPE" != "lora" && "$TUNER_TYPE" != "full" ]]; then
    echo "Error: --tuner must be 'lora' or 'full', got '$TUNER_TYPE'"
    exit 1
fi

if [[ "$MODEL_SIZE" != "2B" && "$MODEL_SIZE" != "4B" && "$MODEL_SIZE" != "8B" ]]; then
    echo "Error: --model_size must be '2B', '4B', or '8B', got '$MODEL_SIZE'"
    exit 1
fi

# Apply defaults based on tuner type if not explicitly set
if [ -z "$FREEZE_VIT" ]; then
    if [ "$TUNER_TYPE" = "lora" ]; then FREEZE_VIT="true"; else FREEZE_VIT="false"; fi
fi
if [ -z "$FREEZE_ALN" ]; then
    if [ "$TUNER_TYPE" = "lora" ]; then FREEZE_ALN="true"; else FREEZE_ALN="false"; fi
fi

# ── Environment setup ──────────────────────────────────────────────
ROOT="/apdcephfs_sgfd/share_303967936/cglin"
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"

# ── Resolve model path ─────────────────────────────────────────────
MODEL_BASE="$HF_HOME/hub/models--Qwen--Qwen3-VL-${MODEL_SIZE}-Instruct"
SNAPSHOT_DIR="$MODEL_BASE/snapshots"
if [ -d "$SNAPSHOT_DIR" ]; then
    MODEL_PATH="$SNAPSHOT_DIR/$(ls "$SNAPSHOT_DIR" | head -1)"
else
    MODEL_PATH="$MODEL_BASE"
fi

# ── Multi-GPU ──────────────────────────────────────────────────────
# ms-swift uses NPROC_PER_NODE env var (not CLI arg) for multi-GPU
if [ "$NUM_GPUS" -gt 1 ]; then
    export NPROC_PER_NODE="$NUM_GPUS"
fi

# ── Tuner-specific arguments ───────────────────────────────────────
TUNER_ARGS=()
if [ "$TUNER_TYPE" = "lora" ]; then
    TUNER_ARGS+=(
        --tuner_type lora
        --lora_rank "$LORA_RANK"
        --lora_alpha $(( LORA_RANK * 2 ))
        --target_modules all-linear
        --learning_rate "${LR_OVERRIDE:-1e-4}"
    )
else
    # Full fine-tuning: lower LR for stability
    TUNER_ARGS=(
        --tuner_type full
        --learning_rate "${LR_OVERRIDE:-1e-5}"
    )
fi

# ── Run swift sft ──────────────────────────────────────────────────
echo "Model: Qwen3-VL-$MODEL_SIZE ($MODEL_PATH)"
echo "Data: $DATA_JSONL"
echo "Output: $OUTPUT_DIR"
echo "Tuner: $TUNER_TYPE"
echo "freeze_vit: $FREEZE_VIT, freeze_aligner: $FREEZE_ALN"
echo "GPUs: $NUM_GPUS"
echo "Epochs: $NUM_EPOCHS, grad_accum: $GRAD_ACCUM, save_steps: $SAVE_STEPS, max_steps: ${MAX_STEPS:-auto}"

# Build optional args
EXTRA_ARGS=()
if [ -n "$MAX_STEPS" ]; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi

# NOTE: no --system flag -- each sample carries its own system prompt
# in messages[0], enabling multi-task training (prediction + judge).
MAX_PIXELS=$((256 * 28 * 28)) \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATA_JSONL" \
    "${TUNER_ARGS[@]}" \
    --freeze_vit "$FREEZE_VIT" \
    --freeze_aligner "$FREEZE_ALN" \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --gradient_checkpointing true \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 3 \
    --logging_steps 10 \
    --check_model false \
    "${EXTRA_ARGS[@]}"

echo "Fine-tuning complete. Checkpoints saved to: $OUTPUT_DIR"
