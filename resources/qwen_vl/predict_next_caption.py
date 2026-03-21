"""
Predict the caption of the next video clip given the current clip's caption
and optionally some frames (images) or the full video of the current clip.

Usage examples:
    # Text-only: predict next caption from current caption alone
    python resources/qwen-vl/predict_next_caption.py \
        --caption "A person walks into a kitchen and opens the fridge." \
        --model_size 8B

    # With a few sampled frames from the current clip
    python resources/qwen-vl/predict_next_caption.py \
        --caption "A person walks into a kitchen and opens the fridge." \
        --images frame_001.jpg frame_005.jpg frame_010.jpg \
        --model_size 4B

    # With a full video file of the current clip
    python resources/qwen-vl/predict_next_caption.py \
        --caption "A drone flies over a mountain valley." \
        --video current_clip.mp4 \
        --model_size 2B

    # Batch mode with a JSONL file
    python resources/qwen-vl/predict_next_caption.py \
        --batch_file inputs.jsonl \
        --model_size 8B \
        --output_file predictions.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


# ──────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────

ROOT = os.environ.get("ROOT", "/apdcephfs_sgfd/share_303967936/cglin")
HF_CACHE = os.path.join(ROOT, ".cache/huggingface/hub")

MODEL_PATHS = {
    "2B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-2B-Instruct"),
    "4B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-4B-Instruct"),
    "8B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-8B-Instruct"),
}


# ──────────────────────────────────────────────────────────────────────
# Prompt design
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a video understanding and prediction assistant. "
    "Given a description (caption) of the current video clip — and optionally "
    "some visual frames or the video itself — your task is to predict what will "
    "most likely happen in the NEXT video clip. "
    "Output ONLY the predicted caption for the next clip. "
    "The caption should be a single, concise paragraph (1-3 sentences) "
    "describing the visual content, actions, and scene of the next clip. "
    "Do NOT repeat the current caption; predict what comes next."
)

def build_user_content(
    caption: str,
    images: Optional[list[str]] = None,
    video: Optional[str] = None,
    sample_fps: int = 2,
    max_frames: int = 64,
):
    """Build the user message content list for the Qwen3-VL chat format."""
    content = []

    # Visual input: video takes priority over images
    if video is not None:
        content.append({
            "video": video,
            "max_frames": max_frames,
            "sample_fps": sample_fps,
            "total_pixels": 20480 * 32 * 32,
            "min_pixels": 64 * 32 * 32,
        })
    elif images is not None and len(images) > 0:
        # Multiple images as interleaved image inputs
        for img_path in images:
            content.append({
                "type": "image",
                "image": img_path,
                "min_pixels": 64 * 28 * 28,
                "max_pixels": 512 * 28 * 28,
            })

    # Text prompt
    if video is not None or (images is not None and len(images) > 0):
        text = (
            f"The caption of this current video clip is:\n\"{caption}\"\n\n"
            "Based on the visual content shown above and the caption, "
            "predict and write the caption for the NEXT video clip "
            "(what will happen next). Output ONLY the predicted caption."
        )
    else:
        text = (
            f"The caption of the current video clip is:\n\"{caption}\"\n\n"
            "Based on this caption, predict and write the caption for the "
            "NEXT video clip (what will most likely happen next). "
            "Output ONLY the predicted caption."
        )

    content.append({"type": "text", "text": text})
    return content


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def resolve_model_path(model_size: str) -> str:
    """Resolve model path, checking for snapshot symlink."""
    base = MODEL_PATHS[model_size]
    snapshot_dir = os.path.join(base, "snapshots")
    if os.path.isdir(snapshot_dir):
        # Use the first (usually only) snapshot
        snapshots = os.listdir(snapshot_dir)
        if snapshots:
            return os.path.join(snapshot_dir, snapshots[0])
    return base


def load_model(model_size: str = "8B", device: str = "cuda"):
    """Load Qwen3-VL model and processor."""
    model_path = resolve_model_path(model_size)
    print(f"Loading Qwen3-VL-{model_size} from {model_path} ...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {device}.")
    return model, processor


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_caption(
    model,
    processor,
    caption: str,
    images: Optional[list[str]] = None,
    video: Optional[str] = None,
    max_new_tokens: int = 256,
    sample_fps: int = 2,
    max_frames: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> str:
    """
    Predict the next video clip's caption.

    Args:
        model: Loaded Qwen3-VL model.
        processor: Loaded Qwen3-VL processor.
        caption: Caption of the current video clip.
        images: Optional list of image file paths (frames from the current clip).
        video: Optional video file path of the current clip.
        max_new_tokens: Max generation length.
        sample_fps: FPS for video sampling.
        max_frames: Max frames to sample from video.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        enable_thinking: Whether to enable Qwen3's thinking/reasoning mode.

    Returns:
        Predicted caption string for the next clip.
    """
    user_content = build_user_content(
        caption=caption,
        images=images,
        video=video,
        sample_fps=sample_fps,
        max_frames=max_frames,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Process vision info
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
    )

    # Only pass `video_kwargs` when there are actual video inputs
    processor_kwargs = {}
    if video_inputs is not None:
        processor_kwargs.update(video_kwargs)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **processor_kwargs,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )

    # Decode only newly generated tokens
    generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
    output_text = processor.decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )

    # If thinking mode is on, strip the thinking part (between <think>...</think>)
    if enable_thinking and "</think>" in output_text:
        output_text = output_text.split("</think>")[-1].strip()

    return output_text.strip()


# ──────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────

def process_batch(
    model,
    processor,
    batch_file: str,
    output_file: str,
    **kwargs,
):
    """
    Process a JSONL file where each line is:
        {"caption": "...", "images": ["path1", ...], "video": "path_or_null"}

    Writes predictions to output JSONL.
    """
    results = []
    with open(batch_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        item = json.loads(line.strip())
        caption = item["caption"]
        images = item.get("images", None)
        video_path = item.get("video", None)

        print(f"[{i+1}/{len(lines)}] Predicting next caption ...")
        pred = predict_next_caption(
            model, processor,
            caption=caption,
            images=images,
            video=video_path,
            **kwargs,
        )
        result = {
            "input_caption": caption,
            "predicted_next_caption": pred,
        }
        results.append(result)
        print(f"  Input:  {caption}")
        print(f"  Output: {pred}\n")

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results written to {output_file}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict the next video clip caption using Qwen3-VL."
    )
    parser.add_argument("--caption", type=str, help="Caption of the current video clip.")
    parser.add_argument("--images", nargs="*", default=None, help="Paths to frame images from the current clip.")
    parser.add_argument("--video", type=str, default=None, help="Path to the current video clip file.")
    parser.add_argument("--model_size", type=str, default="8B", choices=["2B", "4B", "8B"],
                        help="Qwen3-VL model size. Default: 8B.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load model on.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    parser.add_argument("--sample_fps", type=int, default=2, help="FPS for video frame sampling.")
    parser.add_argument("--max_frames", type=int, default=64, help="Max frames to sample from video.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking/reasoning mode.")

    # Batch mode
    parser.add_argument("--batch_file", type=str, default=None,
                        help="JSONL file for batch processing. Each line: {\"caption\": ..., \"images\": [...], \"video\": ...}")
    parser.add_argument("--output_file", type=str, default="predictions.jsonl",
                        help="Output JSONL file for batch results.")

    args = parser.parse_args()

    # Validate inputs
    if args.batch_file is None and args.caption is None:
        parser.error("Either --caption or --batch_file must be provided.")

    # Load model
    model, processor = load_model(args.model_size, args.device)

    if args.batch_file is not None:
        process_batch(
            model, processor,
            batch_file=args.batch_file,
            output_file=args.output_file,
            max_new_tokens=args.max_new_tokens,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        )
    else:
        pred = predict_next_caption(
            model, processor,
            caption=args.caption,
            images=args.images,
            video=args.video,
            max_new_tokens=args.max_new_tokens,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        )
        print("\n" + "=" * 60)
        print("Input caption:")
        print(f"  {args.caption}")
        print("Predicted next caption:")
        print(f"  {pred}")
        print("=" * 60)


if __name__ == "__main__":
    main()
