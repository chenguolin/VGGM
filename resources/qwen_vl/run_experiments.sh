#!/bin/bash
# Ablation experiments for Qwen3-VL structured caption SFT.
#
# Runs small-scale experiments (5k UIDs, 1 epoch) across different configs
# on 8x H20 GPUs, then evaluates each checkpoint.
#
# Usage:
#   bash resources/qwen_vl/run_experiments.sh [--data_only] [--train_only] [--eval_only]
#
# Total estimated time: ~2 hours (data prep: 15 min, training: 70 min, eval: 35 min)

set -euo pipefail

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

# ── Configuration ─────────────────────────────────────────────────
NUM_UIDS=5000           # UIDs for ablation (small scale)
NUM_GPUS=8
NUM_EVAL_SAMPLES=50     # samples for evaluation
NUM_EVAL_CLIPS=4
NUM_WORKERS=8
ABLATION_EPOCHS=1       # quick ablation: 1 epoch
ABLATION_GRAD_ACCUM=16  # effective batch = 8 GPUs * 1 * 16 = 128

BASE_DIR="$PROJECT_ROOT/out/vlm_sft"
DATA_DIR="$BASE_DIR/data"
TRAIN_DIR="$BASE_DIR/ablation"
EVAL_DIR="$BASE_DIR/eval"

mkdir -p "$DATA_DIR" "$TRAIN_DIR" "$EVAL_DIR"

# ── Parse flags ───────────────────────────────────────────────────
DO_DATA=true
DO_TRAIN=true
DO_EVAL=true

while [ $# -gt 0 ]; do
    case "$1" in
        --data_only)  DO_TRAIN=false; DO_EVAL=false; shift ;;
        --train_only) DO_DATA=false; DO_EVAL=false; shift ;;
        --eval_only)  DO_DATA=false; DO_TRAIN=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Activate conda ────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate qa

# ── Utility functions ─────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

prepare_data() {
    local name="$1"
    local frame_strategy="$2"
    local history_vision="$3"
    local output="$DATA_DIR/$name"

    if [ -f "$output/train.jsonl" ]; then
        log "Data $name already exists, skipping."
        return
    fi

    log "Preparing data: $name (frames=$frame_strategy, history=$history_vision, UIDs=$NUM_UIDS)"
    python resources/qwen_vl/prepare_sft_data.py \
        --output_dir "$output" \
        --training_split \
        --frame_strategy "$frame_strategy" \
        --history_vision "$history_vision" \
        --max_samples "$NUM_UIDS" \
        --num_workers "$NUM_WORKERS"

    local n_samples
    n_samples=$(wc -l < "$output/train.jsonl")
    log "Data $name: $n_samples samples"
}

run_train() {
    local exp_name="$1"
    local data_name="$2"
    local model_size="$3"
    local tuner="$4"
    shift 4
    local extra_args=("$@")

    local output="$TRAIN_DIR/$exp_name"

    # Check if already trained (look for any checkpoint)
    if ls "$output"/v*-*/checkpoint-* 1>/dev/null 2>&1; then
        log "Experiment $exp_name already trained, skipping."
        return
    fi

    local data_jsonl="$DATA_DIR/$data_name/train.jsonl"
    if [ ! -f "$data_jsonl" ]; then
        log "ERROR: Data not found: $data_jsonl"
        return 1
    fi

    log "Training: $exp_name (model=$model_size, tuner=$tuner, data=$data_name)"
    bash resources/qwen_vl/finetune.sh \
        "$data_jsonl" "$output" \
        --model_size "$model_size" \
        --tuner "$tuner" \
        --num_gpus "$NUM_GPUS" \
        --epochs "$ABLATION_EPOCHS" \
        --grad_accum "$ABLATION_GRAD_ACCUM" \
        --save_steps 9999 \
        "${extra_args[@]}" 2>&1 | tee "$output.log"

    log "Experiment $exp_name complete."
}

find_checkpoint() {
    # Find the last checkpoint directory for an experiment
    local exp_dir="$1"
    local ckpt
    ckpt=$(find "$exp_dir" -name "checkpoint-*" -type d 2>/dev/null | sort -t- -k2 -n | tail -1)
    echo "$ckpt"
}

run_eval() {
    local exp_name="$1"
    local model_size="$2"
    local frame_strategy="$3"
    local history_vision="$4"

    local exp_dir="$TRAIN_DIR/$exp_name"
    local ckpt
    ckpt=$(find_checkpoint "$exp_dir")

    if [ -z "$ckpt" ]; then
        log "WARNING: No checkpoint found for $exp_name, skipping eval."
        return
    fi

    local eval_out="$EVAL_DIR/${exp_name}.jsonl"
    if [ -f "$eval_out" ]; then
        log "Eval $exp_name already exists, skipping."
        return
    fi

    log "Evaluating: $exp_name (checkpoint=$ckpt)"
    python resources/qwen_vl/run_with_action_dataset.py \
        --num_clips "$NUM_EVAL_CLIPS" \
        --num_samples "$NUM_EVAL_SAMPLES" \
        --model_size "$model_size" \
        --lora_path "$ckpt" \
        --frame_strategy "$frame_strategy" \
        --history_vision "$history_vision" \
        --eval_judge \
        --output_file "$eval_out" 2>&1 | tee "$EVAL_DIR/${exp_name}_eval.log"
}

# ── Phase 1: Data Preparation ────────────────────────────────────
if $DO_DATA; then
    log "========== Phase 1: Data Preparation =========="
    prepare_data "qwen3vl_fl_all_5k"   "first_last"     "all"
    prepare_data "qwen3vl_fml_all_5k"  "first_mid_last" "all"
    prepare_data "qwen3vl_fl_last_5k"  "first_last"     "last"
    prepare_data "qwen3vl_fl_none_5k"  "first_last"     "none"
    log "Data preparation complete."
fi

# ── Phase 2: Training ────────────────────────────────────────────
if $DO_TRAIN; then
    log "========== Phase 2: Training Ablations =========="

    # Group A: model size comparison (2B vs 4B vs 8B, LoRA, first_last + all)
    run_train "qwen3vl_2b_lora_fl_all"  "qwen3vl_fl_all_5k"  "2B" "lora"
    run_train "qwen3vl_4b_lora_fl_all"  "qwen3vl_fl_all_5k"  "4B" "lora"
    run_train "qwen3vl_8b_lora_fl_all"  "qwen3vl_fl_all_5k"  "8B" "lora"

    # Group B: data config (8B LoRA, varying frame_strategy / history_vision)
    # B1 = A3 (already trained above)
    run_train "qwen3vl_8b_lora_fml_all"  "qwen3vl_fml_all_5k"  "8B" "lora"
    run_train "qwen3vl_8b_lora_fl_last"  "qwen3vl_fl_last_5k"  "8B" "lora"
    run_train "qwen3vl_8b_lora_fl_none"  "qwen3vl_fl_none_5k"  "8B" "lora"

    # Group C: tuner comparison (8B full FT with frozen ViT+aligner)
    run_train "qwen3vl_8b_full_fl_all"  "qwen3vl_fl_all_5k"  "8B" "full" \
        --freeze_vit --freeze_aligner

    log "All training complete."
fi

# ── Phase 3: Evaluation ──────────────────────────────────────────
if $DO_EVAL; then
    log "========== Phase 3: Evaluation =========="

    # Also run pretrained baselines (no LoRA) for reference
    for model_size in 2B 8B; do
        local_name="qwen3vl_${model_size,,}_pretrained"
        eval_out="$EVAL_DIR/${local_name}.jsonl"
        if [ ! -f "$eval_out" ]; then
            log "Evaluating pretrained baseline: $model_size"
            python resources/qwen_vl/run_with_action_dataset.py \
                --num_clips "$NUM_EVAL_CLIPS" \
                --num_samples "$NUM_EVAL_SAMPLES" \
                --model_size "$model_size" \
                --frame_strategy "first_last" \
                --history_vision "all" \
                --eval_judge \
                --output_file "$eval_out" 2>&1 | tee "$EVAL_DIR/${local_name}_eval.log"
        fi
    done

    # Evaluate all trained experiments
    run_eval "qwen3vl_2b_lora_fl_all"   "2B" "first_last"     "all"
    run_eval "qwen3vl_4b_lora_fl_all"   "4B" "first_last"     "all"
    run_eval "qwen3vl_8b_lora_fl_all"   "8B" "first_last"     "all"
    run_eval "qwen3vl_8b_lora_fml_all"  "8B" "first_mid_last" "all"
    run_eval "qwen3vl_8b_lora_fl_last"  "8B" "first_last"     "last"
    run_eval "qwen3vl_8b_lora_fl_none"  "8B" "first_last"     "none"
    run_eval "qwen3vl_8b_full_fl_all"   "8B" "first_last"     "all"

    log "All evaluations complete."
fi

# ── Summary ───────────────────────────────────────────────────────
log "========== Summary =========="
SUMMARY="$EVAL_DIR/ablation_summary.txt"
{
    echo "Ablation Experiment Summary ($(date))"
    echo "Model: Qwen3-VL | Data: ${NUM_UIDS} UIDs | Epochs: ${ABLATION_EPOCHS}"
    echo "================================================================"
    echo ""

    for log_file in "$EVAL_DIR"/*_eval.log; do
        [ -f "$log_file" ] || continue
        exp_name=$(basename "$log_file" _eval.log)
        echo "--- $exp_name ---"
        # Extract judge accuracy line
        grep -A5 "Judge Evaluation" "$log_file" 2>/dev/null | head -6 || echo "  (no judge results)"
        # Extract key prediction metrics (segment 1 only for brevity)
        grep -A3 "Segment 1:" "$log_file" 2>/dev/null | head -4 || echo "  (no prediction results)"
        echo ""
    done
} > "$SUMMARY"

cat "$SUMMARY"
log "Summary saved to $SUMMARY"
log "Done!"
