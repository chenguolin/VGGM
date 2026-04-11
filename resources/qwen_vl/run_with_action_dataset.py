"""
Evaluate structured caption prediction and action completion judge on
InternalActionDataset.

Two evaluation modes (can run together):

1. **Prediction**: Given seg0 context (user-provided), autoregressively
   predict seg1, seg2, ... and compare against GT.
2. **Judge**: Given frames from different positions within a segment +
   the intended caption, test if the VLM correctly identifies action
   completion status.

Usage::

    # Prediction only (default)
    python resources/qwen_vl/run_with_action_dataset.py \
        --num_clips 4 --num_samples 20

    # Judge only
    python resources/qwen_vl/run_with_action_dataset.py \
        --num_samples 20 --eval_judge --no_eval_predict

    # Both
    python resources/qwen_vl/run_with_action_dataset.py \
        --num_clips 4 --num_samples 20 --eval_judge
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.options import Options, opt_dict
from src.data.internal_action_dataset import InternalActionDataset
from resources.qwen_vl.structured_caption_predictor import StructuredCaptionPredictor


# ──────────────────────────────────────────────────────────────────────
# Frame I/O helpers
# ──────────────────────────────────────────────────────────────────────

def frames_tensor_to_pil(frames: torch.Tensor) -> list:
    """Convert (F, 3, H, W) float [0,1] tensor to list of PIL images."""
    frames_np = (frames * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(f) for f in frames_np]


def save_frames_as_images(
    frames: torch.Tensor,
    out_dir: str,
    strategy: str = "first_last",
    num_frames: int = 4,
    prefix: str = "frame",
) -> list[str]:
    """Save selected frames to disk and return file paths.

    Args:
        frames: (F, 3, H, W) float [0,1] tensor.
        out_dir: Directory to save images.
        strategy: Frame selection strategy:
            - ``"first_last"``: first and last frame only (2 frames).
            - ``"first_mid_last"``: first, middle, last (3 frames).
            - ``"uniform"``: uniformly sample ``num_frames`` frames.
        num_frames: Number of frames for the ``"uniform"`` strategy.
        prefix: Filename prefix.

    Returns:
        List of saved image file paths.
    """
    pil_images = frames_tensor_to_pil(frames)
    F = len(pil_images)

    if strategy == "first_last":
        indices = [0, F - 1] if F > 1 else [0]
    elif strategy == "first_mid_last":
        if F <= 2:
            indices = list(range(F))
        else:
            indices = [0, F // 2, F - 1]
    elif strategy == "uniform":
        n = min(num_frames, F)
        indices = np.linspace(0, F - 1, n, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown frame strategy: {strategy}")

    paths = []
    for i, idx in enumerate(indices):
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.jpg")
        pil_images[idx].save(path, quality=90)
        paths.append(path)
    return paths


def save_frames_as_video(
    frames: torch.Tensor, out_dir: str, name: str = "clip.mp4", fps: int = 8,
) -> str:
    """Save frames as mp4 video and return file path."""
    import torchvision.io as tio
    video_tensor = (frames * 255).byte().permute(0, 2, 3, 1).cpu()  # (F, H, W, 3) uint8
    path = os.path.join(out_dir, name)
    tio.write_video(path, video_tensor, fps=fps)
    return path


# ──────────────────────────────────────────────────────────────────────
# Text similarity metrics
# ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_n(reference: list[str], hypothesis: list[str], n: int) -> float:
    if len(hypothesis) < n:
        return 0.
    ref_ng = _ngrams(reference, n)
    hyp_ng = _ngrams(hypothesis, n)
    clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
    total = max(sum(hyp_ng.values()), 1)
    return clipped / total


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
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
    """Compute text similarity metrics between GT and prediction."""
    gt_tokens = _tokenize(gt)
    pred_tokens = _tokenize(pred)

    if len(gt_tokens) == 0 or len(pred_tokens) == 0:
        return {"bleu1": 0., "bleu4": 0., "rouge_l": 0., "word_overlap": 0.}

    bleu1 = _bleu_n(gt_tokens, pred_tokens, 1)
    bleu4 = _bleu_n(gt_tokens, pred_tokens, 4)

    lcs = _lcs_length(gt_tokens, pred_tokens)
    rouge_p = lcs / len(pred_tokens)
    rouge_r = lcs / len(gt_tokens)
    rouge_l = 2 * rouge_p * rouge_r / (rouge_p + rouge_r) if (rouge_p + rouge_r) > 0 else 0.

    gt_set, pred_set = set(gt_tokens), set(pred_tokens)
    word_overlap = len(gt_set & pred_set) / len(gt_set | pred_set) if (gt_set | pred_set) else 0.

    return {
        "bleu1": round(bleu1, 4),
        "bleu4": round(bleu4, 4),
        "rouge_l": round(rouge_l, 4),
        "word_overlap": round(word_overlap, 4),
    }


# ──────────────────────────────────────────────────────────────────────
# Judge evaluation helpers
# ──────────────────────────────────────────────────────────────────────

def save_frames_at_position(
    frames: torch.Tensor,
    out_dir: str,
    position: str,
    num_frames: int = 2,
    prefix: str = "judge",
) -> list[str]:
    """Save frames from a specific position within a clip tensor.

    Args:
        frames: (F, 3, H, W) float [0,1] tensor.
        position: ``"end"``, ``"mid"``, or ``"early"``.
        num_frames: Number of frames to save.
        prefix: Filename prefix.

    Returns:
        List of saved image file paths, or empty list if not enough frames.
    """
    pil_images = frames_tensor_to_pil(frames)
    F = len(pil_images)
    if F < 2:
        return []

    if position == "end":
        start = max(0, F - num_frames)
        indices = list(range(start, F))
    elif position == "mid":
        mid_start = int(F * 0.3)
        mid_end = int(F * 0.7)
        if mid_end <= mid_start:
            return []
        n = min(num_frames, mid_end - mid_start)
        indices = np.linspace(mid_start, mid_end - 1, n, dtype=int).tolist()
    elif position == "early":
        end = min(max(int(F * 0.3), num_frames), F)
        n = min(num_frames, end)
        indices = np.linspace(0, end - 1, n, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown position: {position}")

    paths = []
    for i, idx in enumerate(indices):
        path = os.path.join(out_dir, f"{prefix}_{position}_{i:02d}.jpg")
        pil_images[idx].save(path, quality=90)
        paths.append(path)
    return paths


# GT labels for each position
JUDGE_GT_LABELS = {
    "end": {"verdict": "stop"},
    "mid": {"verdict": "continue"},
    "early": {"verdict": "continue"},
}


def evaluate_judge_sample(
    predictor: StructuredCaptionPredictor,
    sample: dict,
    tmp_dir: str,
    temperature: float,
    enable_thinking: bool,
) -> list[dict]:
    """Run judge evaluation on one sample.

    For each segment, test end/mid/early positions and compare VLM's
    verdict against the expected GT label.

    Returns a list of per-test results.
    """
    uid = sample["uid"]
    gt_control_agent = sample.get("control_agent", "")

    prompts = sample["prompt"]
    if isinstance(prompts, str):
        gt_captions_abs = [sample.get("captions_abs", "")]
        gt_caption_deltas = [sample.get("caption_deltas", "")]
        gt_action_labels = [sample.get("action_labels", "")]
        gt_end_states = [sample.get("end_states", "")]
        clip_images = [sample.get("image")]
    else:
        num_clips = len(prompts)
        gt_captions_abs = sample.get("captions_abs", [""] * num_clips)
        gt_caption_deltas = sample.get("caption_deltas", [""] * num_clips)
        gt_action_labels = sample.get("action_labels", [""] * num_clips)
        gt_end_states = sample.get("end_states", [""] * num_clips)
        clip_images = sample.get("image", [None] * num_clips)

    sample_dir = os.path.join(tmp_dir, f"judge_{uid}")
    os.makedirs(sample_dir, exist_ok=True)

    results = []
    for ci, frames in enumerate(clip_images):
        if frames is None or frames.numel() == 0 or frames.shape[0] < 4:
            continue  # need enough frames to sample different positions

        # Judge only needs `end_state`
        intended = {
            "end_state": gt_end_states[ci] if ci < len(gt_end_states) else "",
        }

        for position, gt_label in JUDGE_GT_LABELS.items():
            seg_dir = os.path.join(sample_dir, f"seg{ci}_{position}")
            os.makedirs(seg_dir, exist_ok=True)
            paths = save_frames_at_position(frames, seg_dir, position)
            if not paths:
                continue

            try:
                pred = predictor.judge(
                    images=paths,
                    intended_caption=intended,
                    control_agent=gt_control_agent,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                print(f"    Judge failed for uid={uid} seg{ci} {position}: {e}")
                continue

            gt_verdict = gt_label["verdict"]
            pred_verdict = pred.get("verdict", "")

            results.append({
                "uid": uid,
                "segment": ci,
                "position": position,
                "gt_verdict": gt_verdict,
                "pred_verdict": pred_verdict,
                "correct": gt_verdict == pred_verdict,
                "reason": pred.get("reason", ""),
            })

    return results


def print_judge_summary(judge_results: list[dict]):
    """Print judge evaluation summary."""
    if not judge_results:
        print("\nNo judge results.")
        return

    print(f"\n{'=' * 70}")
    print(f"Judge Evaluation: {len(judge_results)} tests")
    print(f"{'=' * 70}")

    # Overall accuracy
    correct = sum(1 for r in judge_results if r["correct"])
    print(f"\n  Overall verdict accuracy: {correct}/{len(judge_results)} "
          f"= {correct / len(judge_results):.1%}")

    # Per-position accuracy
    for position in ["end", "mid", "early"]:
        pos_results = [r for r in judge_results if r["position"] == position]
        if not pos_results:
            continue
        pos_correct = sum(1 for r in pos_results if r["correct"])
        gt_v = JUDGE_GT_LABELS[position]["verdict"]
        print(f"    {position:8s} (GT={gt_v:10s}): "
              f"{pos_correct}/{len(pos_results)} = {pos_correct / len(pos_results):.1%}")

    # Confusion: what does VLM predict for each position?
    print(f"\n  Verdict distribution per position:")
    for position in ["end", "mid", "early"]:
        pos_results = [r for r in judge_results if r["position"] == position]
        if not pos_results:
            continue
        counts = Counter(r["pred_verdict"] for r in pos_results)
        dist = ", ".join(f"{v}={c}" for v, c in counts.most_common())
        print(f"    {position:8s}: {dist}")

    # Examples
    print(f"\n{'-' * 70}")
    print("Examples:")
    shown = set()
    for r in judge_results:
        pos = r["position"]
        if pos in shown:
            continue
        shown.add(pos)
        mark = "OK" if r["correct"] else "WRONG"
        print(f"  [{pos}] uid={r['uid']} seg{r['segment']}  "
              f"GT={r['gt_verdict']} Pred={r['pred_verdict']} [{mark}]")
        if r["reason"]:
            print(f"    Reason: {r['reason'][:120]}")

    print(f"\n{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────
# Build dataset options
# ──────────────────────────────────────────────────────────────────────

def build_opt(num_clips: int) -> Options:
    """Build Options for InternalActionDataset."""
    import copy
    # Start from the base `wan2.1_t2v` preset (which has `version_new_action=True`)
    opt = copy.deepcopy(opt_dict["wan2.1_t2v"])
    opt.num_clips = num_clips
    opt.num_clips_test = num_clips
    opt.__post_init__()
    return opt


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

PREDICTED_FIELDS = ["action_label", "caption_delta", "end_state"]
GLOBAL_FIELDS = ["global_caption", "control_agent"]


def evaluate_sample(
    predictor: StructuredCaptionPredictor,
    sample: dict,
    tmp_dir: str,
    frame_strategy: str,
    num_frames_vlm: int,
    use_video: bool,
    history_vision: str,
    history_window: int,
    temperature: float,
    enable_thinking: bool,
) -> dict:
    """Run prediction on one sample and compare against GT."""
    uid = sample["uid"]
    gt_global_caption = sample.get("global_caption", "")
    gt_control_agent = sample.get("control_agent", "")

    # Determine number of clips
    prompts = sample["prompt"]
    if isinstance(prompts, str):
        num_clips = 1
        gt_captions_abs = [sample.get("captions_abs", "")]
        gt_action_labels = [sample.get("action_labels", "")]
        gt_caption_deltas = [sample.get("caption_deltas", "")]
        gt_end_states = [sample.get("end_states", "")]
        clip_images = [sample.get("image")]
    else:
        num_clips = len(prompts)
        gt_captions_abs = sample.get("captions_abs", [""] * num_clips)
        gt_action_labels = sample.get("action_labels", [""] * num_clips)
        gt_caption_deltas = sample.get("caption_deltas", [""] * num_clips)
        gt_end_states = sample.get("end_states", [""] * num_clips)
        clip_images = sample.get("image", [None] * num_clips)

    # Prepare visual inputs for each segment
    sample_dir = os.path.join(tmp_dir, f"sample_{uid}")
    os.makedirs(sample_dir, exist_ok=True)

    images_per_segment = []
    videos_per_segment = []

    for ci in range(num_clips):
        frames = clip_images[ci] if clip_images is not None else None
        if frames is not None and frames.numel() > 0:
            if use_video:
                vid_path = save_frames_as_video(
                    frames, sample_dir, name=f"seg_{ci}.mp4")
                images_per_segment.append(None)
                videos_per_segment.append(vid_path)
            else:
                seg_dir = os.path.join(sample_dir, f"seg_{ci}")
                os.makedirs(seg_dir, exist_ok=True)
                paths = save_frames_as_images(
                    frames, seg_dir, strategy=frame_strategy,
                    num_frames=num_frames_vlm, prefix=f"s{ci}")
                images_per_segment.append(paths)
                videos_per_segment.append(None)
        else:
            images_per_segment.append(None)
            videos_per_segment.append(None)

    # Run prediction — use GT `global_caption`, `control_agent`, and seg0
    # as `initial_context` (these are user-provided in the real T2V
    # pipeline, not predicted by the VLM).
    # Compute per-segment frame counts from image tensors
    seg_frame_counts = []
    for ci in range(num_clips):
        frames = clip_images[ci] if clip_images is not None else None
        if frames is not None and hasattr(frames, "shape") and frames.ndim >= 1:
            seg_frame_counts.append(frames.shape[0])
        else:
            seg_frame_counts.append(0)

    initial_context = {
        "global_caption": gt_global_caption,
        "control_agent": gt_control_agent,
        "segment": {
            "action_label": gt_action_labels[0] if gt_action_labels else "",
            "caption_abs": gt_captions_abs[0] if gt_captions_abs else "",
            "end_state": gt_end_states[0] if gt_end_states else "",
            "estimated_frames": seg_frame_counts[0] if seg_frame_counts else 0,
        },
    }
    prediction = predictor.predict_sequence(
        images_per_segment=images_per_segment if not use_video else None,
        videos_per_segment=videos_per_segment if use_video else None,
        initial_context=initial_context,
        history_vision=history_vision,
        history_window=history_window,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    # Compare per-segment fields (skip seg0 which is user-provided)
    segment_metrics = []
    pred_segments = prediction.get("segments", [])
    for si in range(num_clips):
        seg_result = {}
        if si == 0:
            # Seg0 is user-provided, no prediction to evaluate
            segment_metrics.append(seg_result)
            continue

        pred_seg = pred_segments[si] if si < len(pred_segments) else {}

        # Build GT segment dict for comparison
        gt_seg = {
            "action_label": gt_action_labels[si] if si < len(gt_action_labels) else "",
            "caption_delta": gt_caption_deltas[si] if si < len(gt_caption_deltas) else "",
            "end_state": gt_end_states[si] if si < len(gt_end_states) else "",
        }

        for field in PREDICTED_FIELDS:
            gt_val = gt_seg.get(field, "")
            pred_val = pred_seg.get(field, "")
            if gt_val:  # only compute metrics when GT is non-empty
                seg_result[field] = {
                    "gt": gt_val,
                    "pred": pred_val,
                    "metrics": compute_similarity(gt_val, pred_val),
                }

        segment_metrics.append(seg_result)

    return {
        "uid": uid,
        "num_clips": num_clips,
        "prediction": prediction,
        "segment_metrics": segment_metrics,
    }


def print_summary(results: list[dict]):
    """Print aggregated evaluation summary."""
    n = len(results)
    if n == 0:
        print("No results.")
        return

    print(f"\n{'=' * 70}")
    print(f"Summary: {n} samples evaluated")
    print(f"{'=' * 70}")

    # Per-segment field metrics (seg0 is user-provided, seg1+ are predicted)
    max_segs = max(r["num_clips"] for r in results)
    for si in range(1, max_segs):
        print(f"\n  Segment {si} (predicted):")
        for field in PREDICTED_FIELDS:
            values = {}
            for r in results:
                if si < len(r["segment_metrics"]):
                    seg = r["segment_metrics"][si]
                    if field in seg:
                        for mk, mv in seg[field]["metrics"].items():
                            values.setdefault(mk, []).append(mv)
            if values:
                print(f"    {field}:")
                for mk in ["bleu1", "rouge_l", "word_overlap"]:
                    if mk in values:
                        vals = values[mk]
                        print(f"      {mk:15s}  mean={np.mean(vals):.4f}  "
                              f"std={np.std(vals):.4f}  n={len(vals)}")

    # Example
    if results:
        r = results[0]
        print(f"\n{'-' * 70}")
        print(f"Example (uid={r['uid']}):")
        print(f"{'-' * 70}")
        pred = r["prediction"]
        print(f"  global_caption: {pred.get('global_caption', '')[:120]}...")
        print(f"  control_agent:  {pred.get('control_agent', '')[:120]}...")
        for si, seg in enumerate(pred.get("segments", [])):
            print(f"\n  Segment {si}:")
            print(f"    action_label:  {seg.get('action_label', '')}")
            print(f"    caption_abs:   {seg.get('caption_abs', '')[:100]}...")
            if seg.get("caption_delta"):
                print(f"    caption_delta: {seg.get('caption_delta', '')[:100]}...")
            print(f"    end_state:     {seg.get('end_state', '')}")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate structured caption prediction and judge on InternalActionDataset."
    )
    parser.add_argument("--num_clips", type=int, default=3,
                        help="Number of clips (segments) per sample.")
    parser.add_argument("--model_size", type=str, default="2B",
                        choices=["2B", "4B", "8B",
                                 "3.5-0.8B", "3.5-2B", "3.5-4B", "3.5-9B"],
                        help="Qwen3-VL model size.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to evaluate.")
    parser.add_argument("--frame_strategy", type=str, default="first_last",
                        choices=["first_last", "first_mid_last", "uniform"],
                        help="Frame selection strategy per segment.")
    parser.add_argument("--num_frames_for_vlm", type=int, default=4,
                        help="Frames per segment (only used with --frame_strategy uniform).")
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28,
                        help="Max pixels per image for VLM. "
                             "256*28*28=~200K (~196 tok), 512*28*28=~400K (~392 tok).")
    parser.add_argument("--history_vision", type=str, default="all",
                        choices=["none", "last", "all"],
                        help="Visual history mode: "
                             "'none' = text-only history; "
                             "'last' = only previous segment's key frames; "
                             "'all' = all history segments' key frames.")
    parser.add_argument("--history_window", type=int, default=24,
                        help="Sliding window size for history segments. "
                             "When > 0, only seg0 (sink) + last W segments "
                             "are in full detail; middle segments compressed "
                             "into action-label chain. 0 = no windowing.")
    parser.add_argument("--use_video", action="store_true",
                        help="Pass segments as video files instead of frames.")
    parser.add_argument("--temperature", type=float, default=0.,
                        help="Sampling temperature (0 = greedy).")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3 reasoning mode.")
    parser.add_argument("--output_file", type=str,
                        default="structured_caption_predictions.jsonl",
                        help="Output JSONL file for predictions.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to full fine-tuned model checkpoint "
                             "(overrides --model_size HF cache path).")
    parser.add_argument("--eval_judge", action="store_true",
                        help="Also evaluate the action completion judge task.")
    parser.add_argument("--no_eval_predict", action="store_true",
                        help="Skip prediction evaluation (use with --eval_judge "
                             "to run judge only).")
    args = parser.parse_args()

    np.random.seed(args.seed)
    run_predict = not args.no_eval_predict
    run_judge = args.eval_judge

    if not run_predict and not run_judge:
        print("Nothing to evaluate. Use --eval_judge or remove --no_eval_predict.")
        return

    # Build dataset
    print(f"Loading InternalActionDataset (num_clips={args.num_clips}) ...")
    opt = build_opt(args.num_clips)
    dataset = InternalActionDataset(opt, training=False)
    print(f"Dataset size: {len(dataset)}")

    # Load predictor
    predictor = StructuredCaptionPredictor(
        model_size=args.model_size, device=args.device,
        lora_path=args.lora_path, model_path=args.model_path,
        max_pixels=args.max_pixels,
    )

    # Process samples
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    predict_results = []
    judge_results = []
    with tempfile.TemporaryDirectory(prefix="struct_cap_") as tmp_dir:
        for i, idx in enumerate(sample_indices):
            idx = int(idx)
            print(f"\n[{i + 1}/{num_samples}] idx={idx}")

            try:
                sample = dataset[idx]
            except Exception as e:
                print(f"  Skipping idx={idx}: {e}")
                continue

            # Prediction evaluation
            if run_predict:
                try:
                    result = evaluate_sample(
                        predictor, sample, tmp_dir,
                        args.frame_strategy, args.num_frames_for_vlm,
                        args.use_video, args.history_vision,
                        args.history_window,
                        args.temperature, args.enable_thinking,
                    )
                    predict_results.append(result)
                    print(f"  Predict: uid={result['uid']}, {result['num_clips']} segments OK")
                except Exception as e:
                    print(f"  Predict failed for idx={idx}: {e}")

            # Judge evaluation
            if run_judge:
                try:
                    j_results = evaluate_judge_sample(
                        predictor, sample, tmp_dir,
                        args.temperature, args.enable_thinking,
                    )
                    judge_results.extend(j_results)
                    print(f"  Judge: {len(j_results)} tests")
                except Exception as e:
                    print(f"  Judge failed for idx={idx}: {e}")

    # Write results
    output = {}
    if predict_results:
        output["predict"] = predict_results
    if judge_results:
        output["judge"] = judge_results
    with open(args.output_file, "w") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(f"\nWrote results to {args.output_file}")

    # Print summaries
    if predict_results:
        print_summary(predict_results)
    if judge_results:
        print_judge_summary(judge_results)


if __name__ == "__main__":
    main()
