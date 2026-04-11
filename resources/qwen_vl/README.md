# Structured Caption Prediction for Agentic Video Generation

Qwen3-VL / Qwen3.5 predicts structured per-segment captions that drive a
causal DiT video generator.  The VLM and DiT form an agentic loop: the VLM
plans what happens next, the DiT generates it, and the VLM judges whether
to continue, stop, or regenerate.

## Architecture

```
User Input (T2V):
  global_caption + control_agent + seg0 fields
       |
       v
+----------------------------+
|  VLM: Predict Next         |  <-- SFT task 1: next-segment prediction
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
|  VLM: Judge                |  <-- SFT task 2: end-state judge
|  reached end_state?        |
+-------------+--------------+
              |
     stop / continue / regenerate
              |
              v
         (loop back to Predict Next)
```

## SFT Tasks

Two tasks are mixed into a single JSONL, distinguished by system prompts.
ms-swift trains them jointly in multi-task fashion.

### Task 1: Next-Segment Prediction

Given history segments (text + key frames), predict the next segment's
structured caption.  The VLM does NOT see the next segment's frames — it
must predict purely from context.

#### System prompt

```
You are a video prediction assistant. Given the overall video context,
what happened in previous segments (text + key frames), predict what will
happen in the NEXT segment.

You must output a JSON object with these fields:
- "action_label": A short imperative phrase (5-10 words) predicting the
  main action in the next segment.
- "caption_delta": 1-2 sentences predicting what will CHANGE compared to
  the previous segment (new actions, scene changes, camera movement, etc.).
- "end_state": One sentence predicting the visual state at the end of
  the next segment.
- "estimated_frames": An integer estimating how many video frames (at
  16fps) the next segment will last.

Output ONLY valid JSON. No markdown fences, no extra text.
```

#### User input (example: predicting seg3)

```
Video overview: A warrior fights monsters in a snowy arena...
Main character: A bald muscular man with red tattoos...
Previous segments:
<image><image>
Segment 1 (48 frames): [Strike the enemy with axe] The warrior stands in a snowy
arena, gripping a large battle axe. He charges forward toward a frost
giant... End state: The enemy staggers backward from the axe blow.
<image><image>
Segment 2 (32 frames): [Dodge the enemy counterattack] The frost giant swings its
arm, and the warrior rolls sideways to evade. End state: The warrior
crouches at a safe distance, axe ready.
<image><image>
Segment 3 (64 frames): [Leap and deliver an overhead strike] The warrior jumps high
and brings the axe down. End state: The axe is embedded in the giant's
shoulder.
Based on the video context and previous segments above, predict what will
happen in the NEXT segment. Output the structured JSON prediction.
```

**Input structure:**
- `Video overview:` = `global_caption` (user-provided, fixed)
- `Main character:` = `control_agent` (user-provided, fixed)
- Segment 1 (= seg0): `(N frames)` + `[action_label]` + **`caption_abs`** + `End state:`
- Segment 2+ (= seg1+): `(N frames)` + `[action_label]` + **`caption_delta`** + `End state:`
- Each segment header includes the frame count (at 16fps) to provide temporal context
- Each segment may include `<image>` placeholders for key frames
  (controlled by `--history_vision`: `all` / `last` / `none`)

#### Expected output

```json
{
  "action_label": "Finish off the wounded frost giant",
  "caption_delta": "The giant falls to its knees. The warrior pulls the axe free.",
  "end_state": "The frost giant lies defeated on the snow-covered ground.",
  "estimated_frames": 64
}
```

### Task 2: End-State Judge

Given frames from a generated video segment and the intended end state,
judge whether the action has been completed.

#### System prompt

```
You are a video generation quality judge. Given frames from a video
segment and the intended end state describing what the scene SHOULD look
like when the action is complete, decide whether to stop, continue, or
regenerate.

You must output a JSON object with exactly two fields, in this order:
- "reason": One sentence explaining your reasoning.
- "verdict": "stop" if the intended end state has been reached and the
  visual quality is acceptable, "continue" if the action is still in
  progress, "regenerate" if the content is wrong or quality is poor.

Output ONLY valid JSON. No markdown fences, no extra text.
```

#### User input

```
<image><image>
Intended end state:
{"end_state": "The frost giant lies defeated on the snow-covered ground."}

Evaluate whether this video segment has reached the intended end state.
```

**Input structure:**
- `<image>` = frames sampled from the generated video segment
- Only `end_state` is provided (no action_label, caption, or control_agent)

#### Expected output

```json
{"reason": "The giant is still standing and fighting back.", "verdict": "continue"}
```

#### Training data construction

Judge training data is auto-constructed from GT:

- **end** frames → `verdict: "stop"` (action complete)
- **mid** frames → `verdict: "continue"` (in progress)
- **early** frames → `verdict: "continue"` (barely started)
- **cross-segment** frames → `verdict: "regenerate"` (wrong action)

### Task Balance

By default `--judge_per_seg 1`, which randomly samples 1 judge variant
per segment, yielding a ~1:1.5 ratio (prediction:judge). Increase to
`--judge_per_seg 2` for more judge data, or `--no_judge` to skip entirely.

## Files


| File                              | Description                                                                              |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `structured_caption_predictor.py` | Core module: `StructuredCaptionPredictor` with `predict_next()` and `predict_sequence()` |
| `prepare_sft_data.py`             | Build ms-swift SFT training data from InternalActionDataset                              |
| `run_with_action_dataset.py`      | Evaluate prediction quality against GT (BLEU, ROUGE-L, etc.)                             |
| `finetune.sh`                     | Launch ms-swift SFT training                                                             |
| `run_experiments.sh`              | Ablation experiment runner (data prep + train + eval)                                    |
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

## Data Preparation

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


| Argument           | Default                             | Description                                                   |
| ------------------ | ----------------------------------- | ------------------------------------------------------------- |
| `--output_dir`     | `/tmp/qwen_vl_sft_structured`       | Output directory for images + JSONL                           |
| `--data_path`      | `video_action_caption_70w_p1.jsonl` | JSONL filename relative to DATAROOT                           |
| `--frame_strategy` | `first_last`                        | `first_last` (2 frames), `first_mid_last` (3), `uniform` (N) |
| `--num_frames`     | `4`                                 | Frames per segment (only for `uniform` strategy)              |
| `--history_vision` | `all`                               | Visual history: `none`, `last`, `all`                         |
| `--judge_per_seg`  | `1`                                 | Judge samples per segment (1-4)                               |
| `--max_segs`       | `8`                                 | Max segments per video                                        |
| `--no_judge`       | off                                 | Skip judge samples entirely                                   |
| `--training_split` | off                                 | Use training split (default: val 5%)                          |
| `--max_samples`    | all                                 | Cap number of UIDs                                            |
| `--num_workers`    | `8`                                 | Parallel workers                                              |


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


| Option             | Default | Description                                 |
| ------------------ | ------- | ------------------------------------------- |
| `--model_size`     | `2B`    | Qwen3-VL: `2B`/`4B`/`8B`; Qwen3.5: `3.5-2B`/`3.5-4B`/`3.5-9B` |
| `--tuner`          | `lora`  | `lora` or `full`                            |
| `--freeze_vit`     | auto    | Freeze ViT (default: true for lora)         |
| `--freeze_aligner` | auto    | Freeze aligner (default: true for lora)     |
| `--num_gpus`       | `1`     | Number of GPUs                              |
| `--epochs`         | `3`     | Number of training epochs                   |
| `--grad_accum`     | `16`    | Gradient accumulation steps                 |
| `--save_steps`     | `500`   | Save checkpoint every N steps               |
| `--max_steps`      | auto    | Override total training steps               |
| `--lr`             | auto    | Override learning rate                      |
| `--lora_rank`      | `16`    | LoRA rank                                   |


### Examples

```bash
# LoRA on Qwen3-VL-2B (default, ~20GB VRAM)
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_lora

# LoRA on Qwen3-VL-8B, 8 GPUs
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_8b \
    --model_size 8B --num_gpus 8

# LoRA on Qwen3.5-9B
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_35_9b \
    --model_size 3.5-9B

# Full fine-tuning on 8B, freeze ViT + aligner
bash resources/qwen_vl/finetune.sh /tmp/qwen_vl_sft/train.jsonl out/ft_full \
    --model_size 8B --tuner full --freeze_vit --freeze_aligner
```

## Evaluation

### run_with_action_dataset.py

```bash
python resources/qwen_vl/run_with_action_dataset.py \
    --num_clips 4 --model_size 2B --num_samples 50 \
    --frame_strategy first_last --history_vision all
```

Evaluation uses GT `global_caption`, `control_agent`, and seg0 as
`initial_context` (matching the T2V pipeline where these are
user-provided).  Only seg1+ predictions are scored.


| Argument           | Default                                | Description                             |
| ------------------ | -------------------------------------- | --------------------------------------- |
| `--num_clips`      | `3`                                    | Segments per sample                     |
| `--model_size`     | `2B`                                   | Model size key                          |
| `--frame_strategy` | `first_last`                           | Frame selection per segment             |
| `--history_vision` | `all`                                  | Visual history mode                     |
| `--max_pixels`     | `200704`                               | Max pixels per image (~196 tokens)      |
| `--temperature`    | `0.0`                                  | Sampling temperature (0 = greedy)       |
| `--lora_path`      | none                                   | Path to LoRA adapter from SFT           |
| `--model_path`     | none                                   | Path to full FT checkpoint              |
| `--output_file`    | `structured_caption_predictions.jsonl` | Output JSONL                            |


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

Qwen3-VL context: 32K tokens. With `max_pixels=256*28*28` (~196
tokens/image) and `first_last` (2 frames/segment):


| Segments | History frames | History text | Total |
| -------- | -------------- | ------------ | ----- |
| 5        | ~1.6K          | ~0.9K        | ~2.8K |
| 10       | ~3.5K          | ~1.8K        | ~5.7K |
| 20       | ~7.4K          | ~3.3K        | ~11K  |


## Models


| Size      | VRAM (LoRA) | VRAM (full) | HF Name                      |
| --------- | ----------- | ----------- | ----------------------------- |
| 2B        | ~20 GB      | ~40 GB      | Qwen3-VL-2B-Instruct         |
| 4B        | ~35 GB      | ~60 GB      | Qwen3-VL-4B-Instruct         |
| 8B        | ~60 GB      | ~90 GB      | Qwen3-VL-8B-Instruct         |
| 3.5-0.8B  | ~15 GB      | ~25 GB      | Qwen3.5-0.8B                 |
| 3.5-2B    | ~25 GB      | ~45 GB      | Qwen3.5-2B                   |
| 3.5-4B    | ~40 GB      | ~65 GB      | Qwen3.5-4B                   |
| 3.5-9B    | ~70 GB      | ~95 GB      | Qwen3.5-9B                   |


## Ablation Results

See `out/vlm_sft/ABLATION_RESULTS.md` for comprehensive results across
Qwen3-VL and Qwen3.5, different model sizes, LoRA vs full FT, frame
strategies, and history vision modes.

**Note:** Previous ablation results used an incorrect eval protocol
(VLM predicted global_caption/control_agent instead of using GT, and
judge used full caption instead of end_state only). Results after the
fix are in `out/vlm_sft/eval_v2/`.
