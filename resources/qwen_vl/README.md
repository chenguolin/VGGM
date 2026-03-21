# Next Caption Prediction — resources/

Predict the caption of the **next** video clip given the current clip's caption and optional visual input (frames or video).

## Files

| File | Description |
|---|---|
| `predict_next_caption.py` | CLI script for single / batch prediction |
| `next_caption_predictor.py` | Importable `NextCaptionPredictor` class |

## Quick Start

### CLI

```bash
conda activate qa

# Text-only
python resources/predict_next_caption.py \
    --caption "A person walks into a kitchen and opens the fridge." \
    --model_size 8B

# With a few frames
python resources/predict_next_caption.py \
    --caption "A person walks into a kitchen and opens the fridge." \
    --images frame_001.jpg frame_005.jpg frame_010.jpg \
    --model_size 4B

# With full video
python resources/predict_next_caption.py \
    --caption "A drone flies over a mountain valley." \
    --video current_clip.mp4 \
    --model_size 2B

# Batch mode (JSONL input)
python resources/predict_next_caption.py \
    --batch_file inputs.jsonl \
    --model_size 8B \
    --output_file predictions.jsonl
```

### Python API

```python
from resources.next_caption_predictor import NextCaptionPredictor

predictor = NextCaptionPredictor(model_size="8B")

# Text-only
caption = predictor.predict("A person walks into a kitchen and opens the fridge.")

# With frames
caption = predictor.predict(
    "A drone flies over a mountain valley.",
    images=["frame_01.jpg", "frame_05.jpg"],
)

# With video
caption = predictor.predict(
    "Cars race around a track under heavy rain.",
    video="clip.mp4",
)
```

## Batch JSONL format

```json
{"caption": "A person walks into a kitchen.", "images": ["f1.jpg", "f2.jpg"], "video": null}
{"caption": "A drone flies over mountains.", "images": null, "video": "clip.mp4"}
{"caption": "Text-only example.", "images": null, "video": null}
```

## Models

| Size | Approx VRAM | Path |
|---|---|---|
| 2B | ~5 GB | `$HF_CACHE/models--Qwen--Qwen3-VL-2B-Instruct` |
| 4B | ~10 GB | `$HF_CACHE/models--Qwen--Qwen3-VL-4B-Instruct` |
| 8B | ~18 GB | `$HF_CACHE/models--Qwen--Qwen3-VL-8B-Instruct` |

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `--top_p` | 0.9 | Nucleus sampling |
| `--max_new_tokens` | 256 | Max output length |
| `--sample_fps` | 2 | Video frame sampling FPS |
| `--max_frames` | 64 | Max frames from video |
| `--enable_thinking` | off | Enable Qwen3 reasoning mode |
