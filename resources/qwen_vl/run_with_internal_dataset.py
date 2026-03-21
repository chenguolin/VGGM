"""
Run next-caption prediction using InternalDataset as the data source.

Supports both single-step (num_clips=2) and multi-step autoregressive
(num_clips=6) prediction. Each step uses the predicted caption from the
previous step + GT frames of the current clip to predict the next clip.

Usage:
    # Single-step (default, same as before)
    python resources/qwen_vl/run_with_internal_dataset.py --version 2sdiff --num_clips 2 --num_samples 10

    # Multi-step autoregressive (6 clips -> predict clips 1..5)
    python resources/qwen_vl/run_with_internal_dataset.py --version action --num_clips 6 --num_samples 5
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.options import Options
from src.data.internal_dataset import InternalDataset
from resources.qwen_vl.next_caption_predictor import NextCaptionPredictor


def frames_tensor_to_pil(frames: torch.Tensor) -> list:
    """Convert (F, 3, H, W) float [0,1] tensor to list of PIL images."""
    frames_np = (frames * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(f) for f in frames_np]


def save_frames_as_images(frames: torch.Tensor, tmp_dir: str, num_frames_vlm: int, prefix: str = "frame") -> list[str]:
    """Save uniformly sampled frames to disk and return file paths."""
    pil_images = frames_tensor_to_pil(frames)
    F = len(pil_images)
    n = min(num_frames_vlm, F)
    indices = np.linspace(0, F - 1, n, dtype=int).tolist()
    paths = []
    for i, idx in enumerate(indices):
        path = os.path.join(tmp_dir, f"{prefix}_{i:04d}.jpg")
        pil_images[idx].save(path, quality=90)
        paths.append(path)
    return paths


def save_frames_as_video(frames: torch.Tensor, tmp_dir: str, name: str = "clip.mp4", fps: int = 8) -> str:
    """Save all frames as an mp4 video and return the file path."""
    import torchvision.io as tio
    video_tensor = (frames * 255).byte().permute(0, 2, 3, 1).cpu()  # (F, H, W, 3) uint8
    path = os.path.join(tmp_dir, name)
    tio.write_video(path, video_tensor, fps=fps)
    return path


VALID_VERSIONS = ("2s35w", "2sdiff", "action")


def build_opt(version: str, num_clips: int) -> Options:
    """Build Options for loading InternalDataset with the specified version."""
    if version not in VALID_VERSIONS:
        raise ValueError(f"Invalid version '{version}', must be one of {VALID_VERSIONS}")
    # Scale `num_input_frames` proportionally to `num_clips`
    num_input_frames = 1 + 16 * num_clips  # ~17 frames per clip
    opt = Options(
        use_internal_dataset=True,
        version_2s35w=(version == "2s35w"),
        version_2sdiff=(version == "2sdiff"),
        version_action=(version == "action"),
        num_clips=num_clips,
        num_input_frames=num_input_frames,
    )
    return opt


# ──────────────────────────────────────────────────────────────────────
# Similarity metrics (GT vs predicted caption)
# ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_n(reference: list[str], hypothesis: list[str], n: int) -> float:
    """Compute modified n-gram precision (single reference)."""
    if len(hypothesis) < n:
        return 0.
    ref_ng = _ngrams(reference, n)
    hyp_ng = _ngrams(hypothesis, n)
    clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
    total = max(sum(hyp_ng.values()), 1)
    return clipped / total


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    # Space-optimized DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def compute_similarity(gt: str, pred: str) -> dict[str, float]:
    """
    Compute lightweight text similarity metrics between ground truth and prediction.

    Returns:
        bleu1: unigram precision
        bleu4: 4-gram precision
        rouge_l: ROUGE-L F1
        word_overlap: Jaccard similarity of word sets
        length_ratio: len(pred) / len(gt)
    """
    gt_tokens = _tokenize(gt)
    pred_tokens = _tokenize(pred)

    if len(gt_tokens) == 0 or len(pred_tokens) == 0:
        return {"bleu1": 0., "bleu4": 0., "rouge_l": 0., "word_overlap": 0., "length_ratio": 0.}

    # BLEU-1 and BLEU-4
    bleu1 = _bleu_n(gt_tokens, pred_tokens, 1)
    bleu4 = _bleu_n(gt_tokens, pred_tokens, 4)

    # ROUGE-L (F1)
    lcs = _lcs_length(gt_tokens, pred_tokens)
    rouge_p = lcs / len(pred_tokens) if pred_tokens else 0.
    rouge_r = lcs / len(gt_tokens) if gt_tokens else 0.
    rouge_l = 2 * rouge_p * rouge_r / (rouge_p + rouge_r) if (rouge_p + rouge_r) > 0 else 0.

    # Word overlap (Jaccard)
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    word_overlap = len(gt_set & pred_set) / len(gt_set | pred_set) if (gt_set | pred_set) else 0.

    # Length ratio
    length_ratio = len(pred_tokens) / len(gt_tokens)

    return {
        "bleu1": round(bleu1, 4),
        "bleu4": round(bleu4, 4),
        "rouge_l": round(rouge_l, 4),
        "word_overlap": round(word_overlap, 4),
        "length_ratio": round(length_ratio, 4),
    }


def print_summary(results: list[dict], num_samples: int, num_clips: int):
    """Print aggregated comparison statistics."""
    n = len(results)
    num_steps = num_clips - 1

    print("\n" + "=" * 70)
    print(f"Summary: {n}/{num_samples} samples, {num_clips} clips ({num_steps} prediction steps)")
    print("=" * 70)

    if n == 0:
        print("No results to summarize.")
        return

    metric_keys = ["bleu1", "bleu4", "rouge_l", "word_overlap", "length_ratio"]

    if num_steps == 1:
        # Single-step: same as before
        for key in metric_keys:
            values = [r["steps"][0][key] for r in results]
            mean, std, median = np.mean(values), np.std(values), np.median(values)
            print(f"  {key:15s}  mean={mean:.4f}  std={std:.4f}  median={median:.4f}  "
                  f"min={min(values):.4f}  max={max(values):.4f}")
    else:
        # Multi-step: show per-step and overall
        for step_idx in range(num_steps):
            print(f"\n  Step {step_idx + 1} (clip {step_idx} -> clip {step_idx + 1}):")
            for key in metric_keys:
                values = [r["steps"][step_idx][key] for r in results if step_idx < len(r["steps"])]
                if not values:
                    continue
                mean = np.mean(values)
                print(f"    {key:15s}  mean={mean:.4f}  std={np.std(values):.4f}  median={np.median(values):.4f}")

        # Overall average across all steps
        print(f"\n  Overall (averaged across all {num_steps} steps):")
        for key in metric_keys:
            all_values = []
            for r in results:
                for s in r["steps"]:
                    all_values.append(s[key])
            if all_values:
                print(f"    {key:15s}  mean={np.mean(all_values):.4f}  std={np.std(all_values):.4f}")

    # Show example comparisons
    print("\n" + "-" * 70)
    print("Example (first sample):")
    print("-" * 70)
    if results:
        r = results[0]
        print(f"  uid={r['uid']}")
        print(f"  Clip 0 (seed): {r['caption_clip0'][:120]}...")
        for s in r["steps"]:
            si = s["step"]
            print(f"\n  --- Step {si + 1}: predict clip {si + 1} ---")
            print(f"  GT:        {s['gt'][:120]}...")
            print(f"  Predicted: {s['predicted'][:120]}...")
            print(f"  BLEU-1={s['bleu1']:.3f}  ROUGE-L={s['rouge_l']:.3f}  overlap={s['word_overlap']:.3f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL next-caption prediction on InternalDataset."
    )
    parser.add_argument("--version", type=str, default="2sdiff", choices=["2s35w", "2sdiff", "action"],
                        help="Dataset version. Default: 2sdiff.")
    parser.add_argument("--num_clips", type=int, default=2,
                        help="Number of clips per sample (2-6). Default: 2.")
    parser.add_argument("--model_size", type=str, default="8B", choices=["2B", "4B", "8B"],
                        help="Model size of Qwen3-VL.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for prediction.")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to process.")
    parser.add_argument("--num_frames_for_vlm", type=int, default=4,
                        help="Frames to sample per clip for VLM.")
    parser.add_argument("--use_video", action="store_true",
                        help="Pass clips as video files instead of frames.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling.")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode for Qwen3-VL.")
    parser.add_argument("--output_file", type=str, default="next_caption_predictions.jsonl",
                        help="File to save predictions.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--training_split", action="store_true",
                        help="Use training split of InternalDataset.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    num_steps = args.num_clips - 1

    # Build dataset
    print(f"Loading InternalDataset (version_{args.version}, num_clips={args.num_clips}) ...")
    opt = build_opt(args.version, args.num_clips)
    dataset = InternalDataset(opt, training=args.training_split)
    print(f"Dataset size: {len(dataset)}")

    # Load VLM
    predictor = NextCaptionPredictor(model_size=args.model_size, device=args.device)

    # Process samples
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    results = []
    with tempfile.TemporaryDirectory(prefix="qwen_vl_frames_") as tmp_dir:
        for i, idx in enumerate(sample_indices):
            idx = int(idx)
            print(f"\n{'='*60}")
            print(f"[{i+1}/{num_samples}] Sample index={idx}")

            try:
                sample = dataset[idx]
            except Exception as e:
                print(f"  Skipping idx={idx}: {e}")
                continue

            uid = sample["uid"]
            prompts = sample["prompt"]  # List[str] with `num_clips` captions

            # Ensure we got multiple clips
            if isinstance(prompts, str):
                print(f"  Skipping uid={uid}: got single caption (need {args.num_clips} clips)")
                continue
            if len(prompts) < args.num_clips:
                print(f"  Skipping uid={uid}: only {len(prompts)} captions (need {args.num_clips})")
                continue

            # Prepare frames for each clip
            sample_dir = os.path.join(tmp_dir, f"sample_{idx}")
            os.makedirs(sample_dir, exist_ok=True)

            images_per_clip = []  # list of (list[str] or None) per clip
            has_images = "image" in sample and sample["image"] is not None

            if has_images:
                clip_frames_list = sample["image"]  # List[Tensor(F_i, 3, H, W)]
                for ci, clip_frames in enumerate(clip_frames_list):
                    if args.use_video:
                        vid_path = save_frames_as_video(clip_frames, sample_dir, name=f"clip_{ci}.mp4")
                        images_per_clip.append(None)  # handled via `videos_per_clip`
                    else:
                        clip_dir = os.path.join(sample_dir, f"clip_{ci}")
                        os.makedirs(clip_dir, exist_ok=True)
                        paths = save_frames_as_images(clip_frames, clip_dir, args.num_frames_for_vlm, prefix=f"c{ci}")
                        images_per_clip.append(paths)

            # Build `videos_per_clip` if using video mode
            videos_per_clip = None
            if args.use_video and has_images:
                videos_per_clip = [os.path.join(sample_dir, f"clip_{ci}.mp4") for ci in range(len(prompts))]
                images_per_clip = [None] * len(prompts)

            print(f"  uid={uid}, {len(prompts)} clips")
            print(f"  Clip 0: {prompts[0][:80]}...")

            # Run prediction
            if num_steps == 1:
                # Single-step: use the simple predict() method
                predicted = predictor.predict(
                    caption=prompts[0],
                    images=images_per_clip[0] if images_per_clip else None,
                    video=videos_per_clip[0] if videos_per_clip else None,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                )
                step_predictions = [predicted]
            else:
                # Multi-step autoregressive
                def step_cb(step, pred):
                    gt = prompts[step + 1] if (step + 1) < len(prompts) else "N/A"
                    print(f"  Step {step + 1}: predict clip {step + 1}")
                    print(f"    GT:   {gt[:100]}...")
                    print(f"    Pred: {pred[:100]}...")

                step_predictions = predictor.predict_sequence(
                    captions=prompts,
                    images_per_clip=images_per_clip if images_per_clip else None,
                    videos_per_clip=videos_per_clip,
                    num_steps=num_steps,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    enable_thinking=args.enable_thinking,
                    step_callback=step_cb,
                )

            # Compute per-step metrics
            steps_data = []
            for si, pred in enumerate(step_predictions):
                gt_caption = prompts[si + 1]
                metrics = compute_similarity(gt_caption, pred)
                steps_data.append({
                    "step": si,
                    "gt": gt_caption,
                    "predicted": pred,
                    **metrics,
                })
                print(f"  Step {si + 1} metrics: BLEU-1={metrics['bleu1']:.3f}  "
                      f"ROUGE-L={metrics['rouge_l']:.3f}  overlap={metrics['word_overlap']:.3f}")

            result = {
                "uid": uid,
                "dataset_index": idx,
                "num_clips": args.num_clips,
                "caption_clip0": prompts[0],
                "gt_captions": prompts[1:],
                "predicted_captions": step_predictions,
                "input_mode": "video" if args.use_video else f"{args.num_frames_for_vlm}_frames",
                "steps": steps_data,
            }
            results.append(result)

    # Write results
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary comparison
    print_summary(results, num_samples, args.num_clips)


if __name__ == "__main__":
    main()
