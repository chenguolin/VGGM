from typing import *

import argparse
import os
import av
import numpy as np
from tqdm import tqdm
import accelerate

from depth_anything_3.api import DepthAnything3


def get_video_subset(mp4_paths, rank, world_size):
    return [mp4 for i, mp4 in enumerate(mp4_paths) if i % world_size == rank]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, required=True, help="Path to the folder containing input .mp4 videos.")
    args = parser.parse_args()

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")

    mp4_paths = [p for p in os.listdir(args.root) if p.endswith(".mp4")]

    accelerator = accelerate.Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    model = accelerator.prepare(model)

    subset_paths = get_video_subset(mp4_paths, rank, world_size)

    for mp4_path in tqdm(subset_paths, ncols=125, desc=f"Rank {rank}"):
        container = av.open(os.path.join(args.root, mp4_path))
        pil_frames = [frame.to_image() for frame in container.decode(video=0)]

        prediction = model.module.inference(
            image=pil_frames,
            use_ray_pose=True,
            process_res=504,
            process_res_method="upper_bound_resize",
        )

        output_dir = f"{args.root}-DA3"
        output_path = os.path.join(output_dir, mp4_path.replace(".mp4", ".npz"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        save_dict = {
            "depth": np.round(prediction.depth, 8).astype(np.float16),  # (F, H, W)
        }
        if prediction.conf is not None:
            save_dict["conf"] = np.round(prediction.conf, 2).astype(np.float16)  # (F, H, W)
        if prediction.extrinsics is not None:
            save_dict["extrinsics"] = prediction.extrinsics.astype(np.float32)  # (F, 3, 4); opencv w2c or colmap format
        if prediction.intrinsics is not None:
            save_dict["intrinsics"] = prediction.intrinsics.astype(np.float32)  # (F, 3, 3)
        np.savez_compressed(output_path, **save_dict)


if __name__ == "__main__":
    main()
