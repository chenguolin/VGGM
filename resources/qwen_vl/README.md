# Structured Caption Prediction for Agentic Video Generation

Qwen3-VL predicts structured per-segment captions that drive a causal DiT video generator. The VLM and DiT form an agentic loop: the VLM plans what happens next, the DiT generates it, and the VLM judges whether to continue, stop, or regenerate.

## Architecture

```
User Input (T2V):
  global_caption + control_agent + seg0 fields
       |
       v
+----------------------------+
|  Qwen3-VL: Predict Next    |  <-- SFT task 1: next-segment prediction
|  seg1 structured caption   |
|  + estimated_frames        |
+-------------+--------------+
              |
              v
+----------------------------+
|  Causal DiT: Generate      |
|  video frames for seg1     |
+-------------+--------------+
              |  VAE decode
              v
+----------------------------+
|  Qwen3-VL: Judge           |  <-- SFT task 2: action completion
|  action_complete? verdict?  |
+-------------+--------------+
              |
     stop / continue / regenerate
              |
              v
         (loop back to Predict Next)
```

## Files


| File                              | Description                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `structured_caption_predictor.py` | Core module: `StructuredCaptionPredictor` with `predict_next()` and `predict_sequence()` |
| `prepare_sft_data.py`             | Build ms-swift SFT training data from InternalActionDataset                              |
| `run_with_action_dataset.py`      | Evaluate prediction quality against GT (BLEU, ROUGE-L, etc.)                             |
| `finetune.sh`                     | Launch ms-swift SFT training                                                             |
| `legacy/`                         | Previous single-task next-caption prediction (deprecated)                                |


## Quick Start

```bash
conda activate qa

# 1. Prepare SFT data (val split, quick test)
python resources/qwen_vl/prepare_sft_data.py \
    --output_dir /tmp/qwen_vl_sft --max_samples 100

# 2. Prepare SFT data (full training split)
python resources/qwen_vl/prepare_sft_data.py \
    --output_dir /tmp/qwen_vl_sft --training_split --num_workers 8

# 3. Fine-tune with ms-swift
bash resources/qwen_vl/finetune.sh \
    /tmp/qwen_vl_sft/train.jsonl out/qwen_vl_sft

# 4. Evaluate
python resources/qwen_vl/run_with_action_dataset.py \
    --num_clips 4 --num_samples 50 --model_size 2B
```

## SFT Tasks

Two tasks are mixed into a single JSONL, distinguished by system prompts.
ms-swift trains them jointly in multi-task fashion.

### Task 1: Next-Segment Prediction

The core task. Given history segments (text + key frames), predict the
next segment's structured caption.

```
Input:  global_caption + control_agent
      + seg0..seg_{i-1} text fields + key frames
Output: seg_i {action_label, caption_abs, caption_delta, end_state, estimated_frames}
```

The VLM does NOT see the next segment's frames -- it must predict purely
from context. This is what drives the causal DiT at inference time.

### Task 2: Action Completion Judge

Given frames from a segment + the intended caption, judge whether the
action has been completed. Used by the DiT at inference time to decide
when to stop generating the current segment.

```
Input:  frames (sampled at different positions) + intended structured caption
Output: {action_complete, end_state_reached, character_consistent,
         scene_coherent, verdict, reason}
```

Training data is auto-constructed from GT:

- **end** frames -> `verdict: "stop"` (action complete)
- **mid** frames -> `verdict: "continue"` (in progress)
- **early** frames -> `verdict: "continue"` (barely started)
- **cross-segment** frames -> `verdict: "regenerate"` (wrong action)

### Task Balance

By default `--judge_per_seg 1`, which randomly samples 1 judge variant
per segment, yielding a ~1:1.5 ratio (prediction:judge). Increase to
`--judge_per_seg 2` for more judge data, or `--no_judge` to skip entirely.

## Data Preparation Details

### prepare_sft_data.py

```bash
python resources/qwen_vl/prepare_sft_data.py \
    --output_dir /tmp/qwen_vl_sft \
    --training_split \
    --frame_strategy first_last \
    --history_vision all \
    --judge_per_seg 1 \
    --max_segs 8 \
    --num_workers 8
```


| Argument           | Default                             | Description                                                  |
| ------------------ | ----------------------------------- | ------------------------------------------------------------ |
| `--output_dir`     | `/tmp/qwen_vl_sft_structured`       | Output directory for images + JSONL                          |
| `--data_path`      | `video_action_caption_70w_p1.jsonl` | JSONL filename relative to DATAROOT                          |
| `--frame_strategy` | `first_last`                        | `first_last` (2 frames), `first_mid_last` (3), `uniform` (N) |
| `--num_frames`     | `4`                                 | Frames per segment (only for `uniform` strategy)             |
| `--history_vision` | `all`                               | Visual history: `none`, `last`, `all`                        |
| `--judge_per_seg`  | `1`                                 | Judge samples per segment (1-4)                              |
| `--max_segs`       | `8`                                 | Max segments per video                                       |
| `--no_judge`       | off                                 | Skip judge samples entirely                                  |
| `--training_split` | off                                 | Use training split (default: val 5%)                         |
| `--max_samples`    | all                                 | Cap number of UIDs                                           |
| `--num_workers`    | `8`                                 | Parallel workers                                             |


### Output Format

ms-swift compatible JSONL. Each line:

```json
{
  "messages": [
    {"role": "system", "content": "...task-specific system prompt..."},
    {"role": "user", "content": "...text with <image> placeholders..."},
    {"role": "assistant", "content": "...structured JSON response..."}
  ],
  "images": ["/abs/path/to/frame1.jpg", "/abs/path/to/frame2.jpg"]
}
```

## Fine-Tuning

### finetune.sh

```bash
bash resources/qwen_vl/finetune.sh <data_jsonl> <output_dir> [options]
```


| Option             | Default | Description                             |
| ------------------ | ------- | --------------------------------------- |
| `--model_size`     | `2B`    | `2B`, `4B`, `8B`                        |
| `--tuner`          | `lora`  | `lora` or `full`                        |
| `--freeze_vit`     | auto    | Freeze ViT (default: true for lora)     |
| `--freeze_aligner` | auto    | Freeze aligner (default: true for lora) |
| `--num_gpus`       | `1`     | Number of GPUs                          |


Key difference from `legacy/finetune.sh`: no global `--system` flag,
so each sample uses its own system prompt from `messages[0]`.

### Examples

```bash
# LoRA on 2B (default, ~20GB VRAM)
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_lora

# LoRA on 8B, 2 GPUs
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_lora_8b \
    --model_size 8B --num_gpus 2

# Full fine-tuning on 2B, freeze ViT + aligner
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_full \
    --tuner full --freeze_vit --freeze_aligner
```

## Evaluation

### run_with_action_dataset.py

```bash
python resources/qwen_vl/run_with_action_dataset.py \
    --num_clips 4 --model_size 2B --num_samples 50 \
    --frame_strategy first_last --history_vision all
```


| Argument           | Default                                | Description                        |
| ------------------ | -------------------------------------- | ---------------------------------- |
| `--num_clips`      | `3`                                    | Segments per sample                |
| `--model_size`     | `2B`                                   | `2B`, `4B`, `8B`                   |
| `--frame_strategy` | `first_last`                           | Frame selection per segment        |
| `--history_vision` | `all`                                  | Visual history mode                |
| `--max_pixels`     | `200704`                               | Max pixels per image (~196 tokens) |
| `--temperature`    | `0.0`                                  | Sampling temperature (0 = greedy)  |
| `--lora_path`      | none                                   | Path to LoRA adapter from SFT      |
| `--output_file`    | `structured_caption_predictions.jsonl` | Output JSONL                       |


## Python API

```python
from resources.qwen_vl.structured_caption_predictor import StructuredCaptionPredictor

predictor = StructuredCaptionPredictor(model_size="8B")

# User provides seg0 context (T2V mode)
initial_context = {
    "global_caption": "A warrior fights monsters in a snowy arena.",
    "control_agent": "A bald muscular man with red tattoos...",
    "segment": {
        "action_label": "Strike the enemy with axe",
        "caption_abs": "In a snowy arena, the warrior swings...",
        "end_state": "The enemy staggers backward.",
    },
}

# Predict next segments autoregressively
result = predictor.predict_sequence(
    images_per_segment=[["seg0_first.jpg", "seg0_last.jpg"], None, None],
    initial_context=initial_context,
    history_vision="all",
)
# result = {"global_caption": ..., "control_agent": ..., "segments": [...]}
```

## Token Budget

Qwen3-VL context: 32K tokens. With `max_pixels=256*28*28` (~196 tokens/image)
and `first_last` (2 frames/segment):


| Segments | History frames | History text | Total |
| -------- | -------------- | ------------ | ----- |
| 5        | ~1.6K          | ~0.9K        | ~2.8K |
| 10       | ~3.5K          | ~1.8K        | ~5.7K |
| 20       | ~7.4K          | ~3.3K        | ~11K  |


Plenty of headroom for long-horizon prediction.

## Models


| Size | VRAM (LoRA) | VRAM (full) | Path                                               |
| ---- | ----------- | ----------- | -------------------------------------------------- |
| 2B   | ~8 GB       | ~12 GB      | `$HF_CACHE/hub/models--Qwen--Qwen3-VL-2B-Instruct` |
| 4B   | ~14 GB      | ~22 GB      | `$HF_CACHE/hub/models--Qwen--Qwen3-VL-4B-Instruct` |
| 8B   | ~22 GB      | ~40 GB      | `$HF_CACHE/hub/models--Qwen--Qwen3-VL-8B-Instruct` |


