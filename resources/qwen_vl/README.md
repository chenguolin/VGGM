# Next Caption Prediction — resources/qwen_vl/

Predict the caption of the **next** video clip given the current clip's caption and optional visual input (frames or video). Supports single-step and multi-step autoregressive prediction.

## Files

| File | Description |
|---|---|
| `next_caption_predictor.py` | Core module: `NextCaptionPredictor` class with `predict()` and `predict_sequence()` |
| `run_with_internal_dataset.py` | Evaluation script: runs prediction on InternalDataset and computes metrics |

## Python API

```python
from resources.qwen_vl.next_caption_predictor import NextCaptionPredictor

predictor = NextCaptionPredictor(model_size="8B")

# Single-step: predict next clip caption
caption = predictor.predict(
    "A person walks into a kitchen and opens the fridge.",
    images=["frame_01.jpg", "frame_05.jpg"],
)

# Multi-step autoregressive: predict clips 1..5
# Each step uses the predicted caption from last step + GT frames of current clip
predictions = predictor.predict_sequence(
    captions=["clip0 caption", "clip1 caption", ...],  # GT captions (only clip 0 is used as seed)
    images_per_clip=[["f0_a.jpg", "f0_b.jpg"], ["f1_a.jpg"], ...],  # GT frames per clip
    num_steps=5,
)
```

## Evaluation on InternalDataset

```bash
conda activate qa

# Single-step (2 clips)
python resources/qwen_vl/run_with_internal_dataset.py \
    --version action --num_clips 2 --model_size 2B --num_samples 50 --temperature 0

# Multi-step autoregressive (6 clips -> predict clips 1..5)
python resources/qwen_vl/run_with_internal_dataset.py \
    --version action --num_clips 6 --model_size 8B --num_samples 50 --temperature 0

# Different dataset versions
python resources/qwen_vl/run_with_internal_dataset.py --version 2sdiff --num_clips 6 --num_samples 50
python resources/qwen_vl/run_with_internal_dataset.py --version 2s35w  --num_clips 6 --num_samples 50
python resources/qwen_vl/run_with_internal_dataset.py --version action --num_clips 6 --num_samples 50
```

### Dataset versions

| Version | Description |
|---|---|
| `2s35w` | 2-second clips, full scene descriptions. Adjacent clips share scene but differ in wording. |
| `2sdiff` | 2-second clips, differential captions. Adjacent clips have minimal textual differences. |
| `action` | Variable-length action segments. Captions focus on actions with causal continuity. |

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--version` | `2sdiff` | Dataset version (`2s35w`, `2sdiff`, `action`) |
| `--num_clips` | `2` | Clips per sample (2 = single-step, 6 = multi-step) |
| `--model_size` | `8B` | Qwen3-VL model size (`2B`, `4B`, `8B`) |
| `--num_samples` | `10` | Number of samples to evaluate |
| `--num_frames_for_vlm` | `4` | Frames sampled per clip for VLM input |
| `--use_video` | off | Pass clips as video files instead of frames |
| `--temperature` | `0.7` | Sampling temperature (0 = greedy, recommended for eval) |
| `--enable_thinking` | off | Enable Qwen3 reasoning mode |
| `--training_split` | off | Use training split (default: validation 5%) |

## Models

| Size | Approx VRAM | Path |
|---|---|---|
| 2B | ~5 GB | `$HF_CACHE/models--Qwen--Qwen3-VL-2B-Instruct` |
| 4B | ~10 GB | `$HF_CACHE/models--Qwen--Qwen3-VL-4B-Instruct` |
| 8B | ~18 GB | `$HF_CACHE/models--Qwen--Qwen3-VL-8B-Instruct` |
