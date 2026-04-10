"""
Prepare fine-tuning data for Qwen3-VL next-caption prediction.

Reads video clips directly (bypassing InternalDataset's heavy camera/depth loading)
and writes an ms-swift compatible JSONL file with saved frame images.

Usage:
    # Val split, 4 frames per clip (default)
    python resources/qwen_vl/prepare_finetune_data.py \
        --version action --output_dir /tmp/qwen_vl_ft_data

    # Training split, 8 workers for speed
    python resources/qwen_vl/prepare_finetune_data.py \
        --version action --output_dir /tmp/qwen_vl_ft_data --training_split --num_workers 8

    # Quick test with 100 samples
    python resources/qwen_vl/prepare_finetune_data.py \
        --version action --output_dir /tmp/qwen_vl_ft_data --max_samples 100
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

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from resources.qwen_vl.next_caption_predictor import SYSTEM_PROMPT
from src.options import DATAROOT


VALID_VERSIONS = ("2s35w", "2sdiff", "action")


def load_caption_data(version: str):
    """Load caption data and return (uid_to_captions, uid_to_segments_or_none)."""
    if version == "action":
        with open(f"{DATAROOT}/videos_action_caption.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        uid_to_data = {}
        for item in data:
            uid = item["filename"]
            result = item["caption_result"]
            captions = [seg["caption"] for seg in result["segments"]]
            segments = result["segments"]  # need start_time/end_time
            uid_to_data[uid] = {"captions": captions, "segments": segments}
        return uid_to_data
    elif version == "2sdiff":
        with open(f"{DATAROOT}/valid_captions_2sdiff.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return {item["raw_id"]: {"captions": [c["caption"] for c in item["long_caption_lst"]], "segments": None}
                for item in data}
    elif version == "2s35w":
        with open(f"{DATAROOT}/valid_captions_2s35w.jsonl", "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return {item["raw_id"]: {"captions": item["long_caption"], "segments": None}
                for item in data}
    else:
        raise ValueError(f"Invalid version '{version}'")


def split_uids(uid_to_data: dict, training: bool):
    """Split UIDs into train/val (95/5) with fixed seed, filter by existing videos."""
    uids = list(uid_to_data.keys())
    indices = np.random.RandomState(seed=42).permutation(len(uids))
    split_idx = int(0.95 * len(uids))
    if training:
        uids = [uids[i] for i in indices[:split_idx]]
    else:
        uids = [uids[i] for i in indices[split_idx:]]

    # Filter by existing videos
    video_dir = os.path.join(DATAROOT, "video")
    existing_videos = set(os.listdir(video_dir))
    uids = [uid for uid in uids if f"{uid}.mp4" in existing_videos]
    return uids


def build_user_text(caption: str, num_images: int) -> str:
    """Build the user message text with image placeholders."""
    image_tags = "<image>" * num_images
    text_parts = [
        f'The caption of this current video clip is:\n"{caption}"',
        "Based on the visual content shown above and the caption, "
        "predict and write the caption for the NEXT video clip "
        "(what will happen next). Output ONLY the predicted caption.",
    ]
    return image_tags + "\n" + "\n\n".join(text_parts)


def extract_clip_frames(video_path: str, start_time: float, end_time: float, num_frames: int) -> list:
    """Read video, sample `num_frames` uniformly from [start_time, end_time], return PIL images."""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = vr.get_avg_fps()
    total = len(vr)

    start_idx = max(0, int(round(start_time * fps)))
    end_idx = min(total, int(round(end_time * fps)))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, total)

    n = min(num_frames, end_idx - start_idx)
    frame_indices = np.linspace(start_idx, end_idx - 1, n, dtype=int).tolist()

    frames = vr.get_batch(frame_indices).asnumpy()  # (N, H, W, 3) uint8
    del vr
    return [Image.fromarray(f) for f in frames]


def process_one_uid(args_tuple):
    """Process a single UID: read video, save frames, return JSONL lines."""
    uid, data, version, num_frames, images_dir, num_clips = args_tuple
    captions = data["captions"]
    segments = data["segments"]

    if len(captions) < 2:
        return []

    video_path = os.path.join(DATAROOT, "video", f"{uid}.mp4")

    # Compute time boundaries for each clip
    if version == "action":
        clip_times = [(seg["start_time"], seg["end_time"]) for seg in segments]
    else:
        # 2-second clips, 0-based
        clip_times = [(i * 2., (i + 1) * 2.) for i in range(len(captions))]

    # For each consecutive pair, pick a random start (or iterate all valid starts)
    # Use fixed num_clips window
    all_clip_idxs = list(range(len(captions)))
    valid_starts = [
        ci for ci in all_clip_idxs
        if ci + num_clips - 1 < len(captions)
    ]
    if not valid_starts:
        return []

    # Pick one random start per uid (reproducible via uid hash)
    rng = np.random.RandomState(hash(uid) % (2**31))
    start_ci = valid_starts[rng.randint(len(valid_starts))]

    lines = []
    try:
        for ci in range(start_ci, start_ci + num_clips - 1):
            caption_current = captions[ci]
            caption_next = captions[ci + 1]
            start_time, end_time = clip_times[ci]

            # Extract and save frames
            pil_images = extract_clip_frames(video_path, start_time, end_time, num_frames)
            image_paths = []
            for fi, img in enumerate(pil_images):
                path = os.path.join(images_dir, f"{uid}_{ci}_{fi}.jpg")
                img.save(path, quality=90)
                image_paths.append(os.path.abspath(path))

            user_text = build_user_text(caption_current, len(image_paths))
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": caption_next},
                ],
                "images": image_paths,
            }
            lines.append(json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        print(f"  Error processing uid={uid}: {e}")
        return []

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ms-swift fine-tuning data for Qwen3-VL next-caption prediction."
    )
    parser.add_argument("--version", type=str, default="action", choices=list(VALID_VERSIONS),
                        help="Dataset version. Default: action.")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Number of frames per clip to save. Default: 4.")
    parser.add_argument("--output_dir", type=str, default="/tmp/qwen_vl_finetune_data",
                        help="Output directory for images and JSONL.")
    parser.add_argument("--training_split", action="store_true",
                        help="Use training split. Default: val split.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on number of samples (for quick testing).")
    parser.add_argument("--num_clips", type=int, default=2,
                        help="Number of clips per sample. Default: 2.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers. Default: 8.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    t0 = time.time()

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load captions (fast, just reads jsonl)
    print(f"Loading captions (version={args.version}) ...")
    uid_to_data = load_caption_data(args.version)
    print(f"  Total UIDs with captions: {len(uid_to_data)}")

    # Split and filter
    split_name = "train" if args.training_split else "val"
    uids = split_uids(uid_to_data, training=args.training_split)
    print(f"  {split_name} split: {len(uids)} UIDs (after filtering)")

    if args.max_samples is not None:
        uids = uids[:args.max_samples]
        print(f"  Capped to {len(uids)} UIDs")

    # Build work items
    work_items = [
        (uid, uid_to_data[uid], args.version, args.num_frames, images_dir, args.num_clips)
        for uid in uids
    ]

    # Process in parallel
    jsonl_path = os.path.join(args.output_dir, "train.jsonl")
    num_written = 0

    print(f"Processing {len(work_items)} UIDs with {args.num_workers} workers ...")
    with open(jsonl_path, "w") as f_out:
        if args.num_workers <= 1:
            for i, item in enumerate(work_items):
                if (i + 1) % 100 == 0 or i == 0:
                    print(f"  [{i+1}/{len(work_items)}] ...")
                lines = process_one_uid(item)
                for line in lines:
                    f_out.write(line + "\n")
                    num_written += 1
        else:
            with Pool(args.num_workers) as pool:
                for i, lines in enumerate(pool.imap(process_one_uid, work_items, chunksize=16)):
                    if (i + 1) % 500 == 0 or i == 0:
                        elapsed = time.time() - t0
                        print(f"  [{i+1}/{len(work_items)}] {num_written} lines, {elapsed:.0f}s elapsed")
                    for line in lines:
                        f_out.write(line + "\n")
                        num_written += 1

    elapsed = time.time() - t0
    print(f"\nDone! Wrote {num_written} samples to {jsonl_path}")
    print(f"  Images dir: {images_dir}")
    print(f"  Time: {elapsed:.1f}s ({len(uids) / max(elapsed, 1):.1f} UIDs/s)")


if __name__ == "__main__":
    main()
