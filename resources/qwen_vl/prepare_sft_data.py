"""
Prepare SFT data for Qwen3-VL structured caption prediction.

Reads from InternalActionDataset and produces ms-swift compatible JSONL
with saved key frames.  Generates two types of training samples:

1. **Next-segment prediction** (seg i → predict seg i+1): history text +
   history frames → {action_label, caption_delta, end_state,
   estimated_frames}
2. **Judge** (action completion): frames at different positions + intended
   caption → {action_complete, end_state_reached, ..., verdict, reason}

All three tasks are mixed into a single JSONL and distinguished by their
system prompts.  ms-swift trains them jointly in a multi-task fashion.

Usage::

    # Val split, first_last frames (default)
    python resources/qwen_vl/prepare_sft_data.py \\
        --output_dir /tmp/qwen_vl_sft_data

    # Training split, 8 workers, first_mid_last strategy
    python resources/qwen_vl/prepare_sft_data.py \\
        --output_dir /tmp/qwen_vl_sft_data --training_split \\
        --frame_strategy first_mid_last --num_workers 8

    # Quick test with 100 samples
    python resources/qwen_vl/prepare_sft_data.py \\
        --output_dir /tmp/qwen_vl_sft_data --max_samples 100

    # Skip judge samples
    python resources/qwen_vl/prepare_sft_data.py \\
        --output_dir /tmp/qwen_vl_sft_data --no_judge
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from resources.qwen_vl.structured_caption_predictor import (
    SYSTEM_PROMPT_NEXT,
    SYSTEM_PROMPT_JUDGE,
)
from src.options import DATAROOT


# ──────────────────────────────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────────────────────────────

def extract_segment_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    strategy: str = "first_last",
    num_frames: int = 4,
) -> list:
    """Extract frames from a video segment using the given strategy.

    Returns list of PIL images.
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)

    start_idx = max(0, int(round(start_time * fps)))
    end_idx = min(total, int(round(end_time * fps)))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, total)

    if strategy == "first_last":
        if end_idx - start_idx <= 1:
            indices = [start_idx]
        else:
            indices = [start_idx, end_idx - 1]
    elif strategy == "first_mid_last":
        if end_idx - start_idx <= 2:
            indices = list(range(start_idx, end_idx))
        else:
            mid = (start_idx + end_idx) // 2
            indices = [start_idx, mid, end_idx - 1]
    elif strategy == "uniform":
        n = min(num_frames, end_idx - start_idx)
        indices = np.linspace(start_idx, end_idx - 1, n, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown frame strategy: {strategy}")

    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
    del vr
    return [Image.fromarray(f) for f in frames]


def save_pil_images(images: list, out_dir: str, prefix: str) -> list[str]:
    """Save PIL images to disk and return absolute paths."""
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.jpg")
        img.save(path, quality=90)
        paths.append(os.path.abspath(path))
    return paths


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_action_data(data_path: str) -> dict:
    """Load JSONL and return {uid: caption_result}."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return {item["filename"]: item["caption_result"] for item in data}


def split_uids(uid_to_data: dict, training: bool) -> list[str]:
    """Train/val split (95/5) with fixed seed, filter by existing videos."""
    uids = list(uid_to_data.keys())
    indices = np.random.RandomState(seed=42).permutation(len(uids))
    split_idx = int(0.95 * len(uids))
    if training:
        uids = [uids[i] for i in indices[:split_idx]]
    else:
        uids = [uids[i] for i in indices[split_idx:]]

    video_dir = os.path.join(DATAROOT, "video")
    existing_videos = set(os.listdir(video_dir))
    uids = [uid for uid in uids if f"{uid}.mp4" in existing_videos]
    return uids


# ──────────────────────────────────────────────────────────────────────
# SFT sample construction
# ──────────────────────────────────────────────────────────────────────

def _estimate_frames(start_time: float, end_time: float, target_fps: float = 16.) -> int:
    """Estimate the number of generation frames for a segment."""
    duration = end_time - start_time
    return max(1, int(round(duration * target_fps)))


def extract_frames_at_position(
    video_path: str,
    start_time: float,
    end_time: float,
    position: str,
    num_frames: int = 2,
) -> list:
    """Extract frames from a specific position within a segment.

    Args:
        position: Where to sample frames:
            - ``"end"``: last ``num_frames`` frames (action completed).
            - ``"mid"``: middle portion (action in progress).
            - ``"early"``: first 30% (action barely started).

    Returns list of PIL images.
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)

    start_idx = max(0, int(round(start_time * fps)))
    end_idx = min(total, int(round(end_time * fps)))
    if end_idx <= start_idx + 1:
        del vr
        return []

    seg_len = end_idx - start_idx

    if position == "end":
        # Last portion of the segment
        pos_start = max(start_idx, end_idx - max(num_frames, int(seg_len * 0.1)))
        pos_end = end_idx
    elif position == "mid":
        # Middle 30%-70%
        pos_start = start_idx + int(seg_len * 0.3)
        pos_end = start_idx + int(seg_len * 0.7)
    elif position == "early":
        # First 0%-30%
        pos_start = start_idx
        pos_end = start_idx + max(int(seg_len * 0.3), num_frames)
    else:
        raise ValueError(f"Unknown position: {position}")

    pos_end = min(pos_end, end_idx)
    if pos_end <= pos_start:
        del vr
        return []

    n = min(num_frames, pos_end - pos_start)
    indices = np.linspace(pos_start, pos_end - 1, n, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    del vr
    return [Image.fromarray(f) for f in frames]


def build_judge_sample(
    control_agent: str,
    seg: dict,
    frame_image_paths: list[str],
    label: dict,
    seg_index: int = 0,
) -> dict:
    """Build an SFT sample for the action completion judge task.

    Args:
        control_agent: Main agent description.
        seg: Segment dict with ``action_label``, ``caption``, etc.
        frame_image_paths: Paths to the frames being judged.
        label: Dict with ``verdict`` and ``reason``.
        seg_index: 0-based segment index (unused after simplification,
            kept for API compatibility).
    """
    image_tags = "".join("<image>" for _ in frame_image_paths)

    # Judge only needs `end_state` to decide stop/continue/regenerate
    intended = json.dumps({
        "end_state": seg["caption"]["end_state"],
    }, ensure_ascii=False)

    user_text = (
        f"{image_tags}\n"
        f"Intended end state:\n{intended}\n\n"
        "Evaluate whether this video segment has reached "
        "the intended end state."
    )

    response = json.dumps(label, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response},
        ],
        "images": frame_image_paths,
    }


def build_next_sample(
    global_caption: str,
    control_agent: str,
    history_segs: list[dict],
    history_image_paths: list[list[str]],
    next_seg: dict,
    estimated_frames: int,
    history_vision: str = "all",
) -> dict:
    """Build an SFT sample for the next-segment prediction task.

    The VLM sees only history segments (text + optional frames) and must
    **predict** the next segment's structured caption.  It does NOT see
    the next segment's frames.

    Args:
        history_segs: List of previous segment dicts (GT).
        history_image_paths: Key frame paths for each history segment.
        next_seg: The target segment to predict (GT, used as label).
        estimated_frames: GT frame count for the target segment.
        history_vision: ``"none"``, ``"last"``, or ``"all"`` — controls which
            history segments include their key frames in the training sample.
    """
    all_image_paths = []  # accumulate in order
    user_parts = []

    # Text preamble
    user_parts.append(
        f"Video overview: {global_caption}\n"
        f"Main character: {control_agent}"
    )

    # History segments: interleave key frames and text
    user_parts.append("Previous segments:")
    for i, seg in enumerate(history_segs):
        # Decide whether to include this segment's frames
        include_frames = False
        if history_vision == "all":
            include_frames = True
        elif history_vision == "last" and i == len(history_segs) - 1:
            include_frames = True

        if include_frames and i < len(history_image_paths) and history_image_paths[i]:
            for path in history_image_paths[i]:
                user_parts.append("<image>")
                all_image_paths.append(path)

        # Seg1 uses `caption_abs` (no prior segment to diff against);
        # seg2+ uses `caption_delta` (describes change from previous seg)
        if i == 0:
            scene = seg["caption"]["caption_abs"]
        else:
            scene = seg["caption"].get("caption_delta", "") or seg["caption"]["caption_abs"]

        seg_desc = (
            f"Segment {i + 1}: [{seg['action_label']}] "
            f"{scene}"
        )
        end_state = seg["caption"].get("end_state", "")
        if end_state:
            seg_desc += f" End state: {end_state}"
        user_parts.append(seg_desc)

    # Instruction: predict the next segment (no next-segment frames given)
    user_parts.append(
        "Based on the video context and previous segments above, "
        "predict what will happen in the NEXT segment. "
        "Output the structured JSON prediction."
    )

    user_text = "\n".join(user_parts)

    # Assistant response: GT for the next segment
    caption_delta = next_seg["caption"].get("caption_delta", "")
    response = json.dumps({
        "action_label": next_seg["action_label"],
        "caption_delta": caption_delta,
        "end_state": next_seg["caption"]["end_state"],
        "estimated_frames": estimated_frames,
    }, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_NEXT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response},
        ],
        "images": all_image_paths,
    }


# ──────────────────────────────────────────────────────────────────────
# Per-UID processing
# ──────────────────────────────────────────────────────────────────────

def process_one_uid(args_tuple):
    """Process a single UID: extract frames, build SFT samples."""
    uid, caption_result, strategy, num_frames, images_dir, history_vision, max_segs, build_judge, judge_per_seg = args_tuple

    global_caption = caption_result.get("global_caption", "")
    control_agent = caption_result.get("control_agent", "")
    segments = caption_result.get("segments", [])

    if len(segments) < 2:
        return {"caption": [], "judge": []}

    video_path = os.path.join(DATAROOT, "video", f"{uid}.mp4")
    if not os.path.exists(video_path):
        return {"caption": [], "judge": []}

    # Limit number of segments to process
    n_segs = min(len(segments), max_segs)

    # Extract and save frames for each segment (using the configured strategy)
    seg_image_paths = []
    try:
        for si in range(n_segs):
            seg = segments[si]
            start_t = float(seg["start_time"])
            end_t = float(seg["end_time"])
            if end_t - start_t < 0.5:
                return {"caption": [], "judge": []}

            pil_images = extract_segment_frames(
                video_path, start_t, end_t, strategy, num_frames)
            paths = save_pil_images(
                pil_images, images_dir, prefix=f"{uid}_s{si}")
            seg_image_paths.append(paths)
    except Exception:
        return {"caption": [], "judge": []}

    caption_lines = []
    judge_lines = []

    # ── Next-segment prediction samples ─────────────────────────────
    for si in range(1, n_segs):
        est_frames = _estimate_frames(
            float(segments[si]["start_time"]), float(segments[si]["end_time"]))
        try:
            sample = build_next_sample(
                global_caption, control_agent,
                history_segs=segments[:si],
                history_image_paths=seg_image_paths[:si],
                next_seg=segments[si],
                estimated_frames=est_frames,
                history_vision=history_vision,
            )
            caption_lines.append(json.dumps(sample, ensure_ascii=False))
        except (KeyError, IndexError):
            continue

    # ── Judge (action completion) samples ───────────────────────────
    # For each segment, randomly sample `judge_per_seg` judge variants
    # from 4 possible types: end(stop), mid(continue), early(continue),
    # cross(regenerate).  This keeps the judge:prediction ratio balanced.
    if build_judge:
        rng = np.random.RandomState(hash(uid) % (2**31))

        # All candidate builders per segment
        judge_types = ["end", "mid", "early"]
        if n_segs >= 2:
            judge_types.append("cross")

        for si in range(n_segs):
            seg = segments[si]
            start_t = float(seg["start_time"])
            end_t = float(seg["end_time"])
            if end_t - start_t < 1.5:
                continue

            # Randomly pick which types to generate for this segment
            chosen = rng.choice(
                judge_types,
                size=min(judge_per_seg, len(judge_types)),
                replace=False,
            ).tolist()

            for jtype in chosen:
                try:
                    if jtype == "end":
                        frames = extract_frames_at_position(
                            video_path, start_t, end_t, "end", num_frames=2)
                        if not frames:
                            continue
                        paths = save_pil_images(
                            frames, images_dir, prefix=f"{uid}_s{si}_judge_end")
                        label = {
                            "reason": "The action has been fully completed and the end state matches the description.",
                            "verdict": "stop",
                        }
                    elif jtype == "mid":
                        frames = extract_frames_at_position(
                            video_path, start_t, end_t, "mid", num_frames=2)
                        if not frames:
                            continue
                        paths = save_pil_images(
                            frames, images_dir, prefix=f"{uid}_s{si}_judge_mid")
                        label = {
                            "reason": "The action is in progress but the end state has not been reached yet.",
                            "verdict": "continue",
                        }
                    elif jtype == "early":
                        frames = extract_frames_at_position(
                            video_path, start_t, end_t, "early", num_frames=2)
                        if not frames:
                            continue
                        paths = save_pil_images(
                            frames, images_dir, prefix=f"{uid}_s{si}_judge_early")
                        label = {
                            "reason": "The action has barely started; the intended action is not yet underway.",
                            "verdict": "continue",
                        }
                    elif jtype == "cross":
                        other_si = rng.choice([j for j in range(n_segs) if j != si])
                        other_seg = segments[other_si]
                        other_start = float(other_seg["start_time"])
                        other_end = float(other_seg["end_time"])
                        frames = extract_frames_at_position(
                            video_path, other_start, other_end, "end", num_frames=2)
                        if not frames:
                            continue
                        paths = save_pil_images(
                            frames, images_dir,
                            prefix=f"{uid}_s{si}_judge_cross{other_si}")
                        label = {
                            "reason": "The frames show a different action from what was intended in the caption.",
                            "verdict": "regenerate",
                        }
                    else:
                        continue

                    sample = build_judge_sample(control_agent, seg, paths, label, seg_index=si)
                    judge_lines.append(json.dumps(sample, ensure_ascii=False))
                except Exception:
                    continue

    return {"caption": caption_lines, "judge": judge_lines}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT data for Qwen3-VL structured caption prediction."
    )
    parser.add_argument("--data_path", type=str,
                        default="video_action_caption_70w_p1.jsonl",
                        help="JSONL filename relative to DATAROOT.")
    parser.add_argument("--output_dir", type=str,
                        default="/tmp/qwen_vl_sft_structured",
                        help="Output directory for images and JSONL.")
    parser.add_argument("--frame_strategy", type=str, default="first_last",
                        choices=["first_last", "first_mid_last", "uniform"],
                        help="Frame selection strategy per segment.")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Frames per segment (only for uniform strategy).")
    parser.add_argument("--history_vision", type=str, default="all",
                        choices=["none", "last", "all"],
                        help="Visual history mode for next-segment samples.")
    parser.add_argument("--max_segs", type=int, default=8,
                        help="Max segments per video to process.")
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip judge (action completion) samples.")
    parser.add_argument("--judge_per_seg", type=int, default=1,
                        help="Number of judge samples per segment (1-4). "
                             "Default 1 keeps judge:prediction ratio ~1:1.")
    parser.add_argument("--training_split", action="store_true",
                        help="Use training split (default: val split).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of UIDs.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel workers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load caption data
    data_path = os.path.join(DATAROOT, args.data_path)
    print(f"Loading captions from {data_path} ...")
    uid_to_data = load_action_data(data_path)
    print(f"  Total UIDs: {len(uid_to_data)}")

    # Split
    split_name = "train" if args.training_split else "val"
    uids = split_uids(uid_to_data, training=args.training_split)
    print(f"  {split_name} split: {len(uids)} UIDs")

    if args.max_samples is not None:
        uids = uids[:args.max_samples]
        print(f"  Capped to {len(uids)} UIDs")

    # Build work items
    build_judge = not args.no_judge
    judge_per_seg = min(args.judge_per_seg, 4)
    work_items = [
        (uid, uid_to_data[uid], args.frame_strategy, args.num_frames,
         images_dir, args.history_vision, args.max_segs, build_judge, judge_per_seg)
        for uid in uids
    ]

    # Process
    jsonl_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
    num_caption, num_judge = 0, 0

    print(f"Processing {len(work_items)} UIDs with {args.num_workers} workers ...")
    print(f"  Tasks: next-segment prediction"
          f"{' + action completion judge' if build_judge else ''}")
    with open(jsonl_path, "w") as f_out:
        if args.num_workers <= 1:
            for i, item in enumerate(work_items):
                if (i + 1) % 100 == 0:
                    print(f"  [{i + 1}/{len(work_items)}] ...")
                result = process_one_uid(item)
                for line in result["caption"]:
                    f_out.write(line + "\n")
                    num_caption += 1
                for line in result["judge"]:
                    f_out.write(line + "\n")
                    num_judge += 1
        else:
            with Pool(args.num_workers) as pool:
                for i, result in enumerate(
                    pool.imap(process_one_uid, work_items, chunksize=16)
                ):
                    if (i + 1) % 500 == 0:
                        elapsed = time.time() - t0
                        total_so_far = num_caption + num_judge
                        print(f"  [{i + 1}/{len(work_items)}] "
                              f"{total_so_far} samples, "
                              f"{elapsed:.0f}s elapsed")
                    for line in result["caption"]:
                        f_out.write(line + "\n")
                        num_caption += 1
                    for line in result["judge"]:
                        f_out.write(line + "\n")
                        num_judge += 1

    elapsed = time.time() - t0
    total = num_caption + num_judge
    print(f"\nDone! Wrote {total} samples to {jsonl_path}")
    print(f"  Next-segment prediction: {num_caption}")
    print(f"  Action completion judge: {num_judge}")
    print(f"  Images dir: {images_dir}")
    print(f"  Time: {elapsed:.1f}s ({len(uids) / max(elapsed, 1):.1f} UIDs/s)")


if __name__ == "__main__":
    main()
