from typing import *

import os
import numpy as np
import torch
from tqdm import tqdm
import accelerate
from PIL import Image
from decord import VideoReader, cpu

from depth_anything_3.api import DepthAnything3

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import DATAROOT


TARGET_FPS = 16.


def get_video_subset(uids, rank, world_size):
    return [uid for i, uid in enumerate(uids) if i % world_size == rank]


def build_16fps_frame_indices(num_frames: int, src_fps: float, target_fps: float = 16.) -> np.ndarray:
    if num_frames <= 0:
        return np.array([], dtype=np.int64)
    if src_fps <= 0:
        return np.arange(num_frames, dtype=np.int64)
    duration = num_frames / src_fps
    target_ts = np.arange(0., duration, 1. / target_fps, dtype=np.float32)
    frame_indices = np.floor(target_ts * src_fps).astype(np.int64)
    frame_indices = np.clip(frame_indices, 0, num_frames - 1)
    return frame_indices


def main():
    accelerator = accelerate.Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device
    torch.cuda.set_device(device)

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device)

    # Get UIDs from InternalDataset structure
    dataset_root = DATAROOT
    caption_dir = os.path.join(dataset_root, "valid_captions")
    uids = [f.replace(".json", "") for f in os.listdir(caption_dir) if f.endswith(".json")]

    subset_uids = get_video_subset(uids, rank, world_size)

    for uid in tqdm(subset_uids, ncols=125, desc=f"Rank {rank}"):
        video_path = os.path.join(dataset_root, "video", f"{uid}.mp4")
        output_root = os.path.join(dataset_root, "da3", uid)
        os.makedirs(output_root, exist_ok=True)

        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames_src, fps_src = len(vr), vr.get_avg_fps()
        frame_idxs_16fps = build_16fps_frame_indices(num_frames_src, fps_src, target_fps=TARGET_FPS)
        num_frames, fps = len(frame_idxs_16fps), TARGET_FPS

        clip_idx = 1  # start from 1
        while True:
            # `5`: hard-coded for 5s-clip; `12`: hard-coded for clip-overlap
            start_frame_idx = int(round((clip_idx - 1) * 5 * fps)) - 12 * (clip_idx - 1)
            end_frame_idx = start_frame_idx + int(round(5 * fps))

            frame_idxs = np.arange(num_frames, dtype=int)
            if start_frame_idx is not None:
                frame_idxs = frame_idxs[frame_idxs >= start_frame_idx]
            if end_frame_idx is not None:
                frame_idxs = frame_idxs[frame_idxs < end_frame_idx]
            if len(frame_idxs) == 0:
                break

            src_frame_idxs = frame_idxs_16fps[frame_idxs]
            frames = vr.get_batch(src_frame_idxs.tolist()).asnumpy()
            pil_frames = [Image.fromarray(frame) for frame in frames]

            # DA3 inference
            prediction = model.inference(
                image=pil_frames,
                use_ray_pose=True,
                process_res=504,
                process_res_method="upper_bound_resize",
            )

            segment_dir = os.path.join(output_root, f"{clip_idx:02d}")
            os.makedirs(segment_dir, exist_ok=True)

            save_dict = {
                "depth": np.round(prediction.depth, 8).astype(np.float16),  # (F, H, W)
            }
            if prediction.conf is not None:
                save_dict["conf"] = np.round(prediction.conf, 2).astype(np.float16)  # (F, H, W)
            if prediction.extrinsics is not None:
                save_dict["extrinsics"] = prediction.extrinsics.astype(np.float32)  # (F, 3, 4); opencv w2c or colmap format
            if prediction.intrinsics is not None:
                save_dict["intrinsics"] = prediction.intrinsics.astype(np.float32)  # (F, 3, 3)
            for key, value in save_dict.items():
                np.save(os.path.join(segment_dir, f"{key}.npy"), value)

            clip_idx += 1


if __name__ == "__main__":
    main()
