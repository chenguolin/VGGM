from typing import *

import os
import numpy as np
import json
from decord import VideoReader, cpu
import torch
import torchvision.transforms as tvT

from src.options import Options
from src.data.base_dataset import BaseDataset
from src.utils.geo_util import inverse_c2w, intrinsics_to_fxfycxcy, unproject_depth


class InternalDataset(BaseDataset):
    # Hard-coded constants for the internal dataset
    DA3_FPS = 16.
    CLIP_SECONDS = 5.
    CLIP_OVERLAP_FRAMES = 12

    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "internal", training)

        uids = os.listdir(f"{self.root}/valid_captions")
        indices = np.random.RandomState(seed=42).permutation(len(uids))
        if training:
            self.uids = [uids[i].strip(".json") for i in indices[:int(0.95 * len(uids))]]
        else:
            self.uids = [uids[i].strip(".json") for i in indices[int(0.95 * len(uids)):]]

        self.valid_idxs = list(range(len(self.uids)))

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        _num_tries = max(10, len(self.uids))
        for _ in range(_num_tries):
            uid = self.uids[idx]
            with open(f"{self.root}/valid_captions/{uid}.json", "r", encoding="utf-8") as f:
                all_captions = json.load(f)  # Dict[str, str]: clip_idx -> long caption
            dataset_source = "Internal"

            clip_indices = [int(k) for k in all_captions.keys()]
            clip_idx_set = set(clip_indices)
            num_clips = max(1, int(self.opt.num_clips))
            valid_start_clip_idxs = [
                clip_idx for clip_idx in clip_indices
                if all((clip_idx + offset) in clip_idx_set for offset in range(num_clips))
            ]

            # Ensure selected clips are consecutive and all have captions
            if len(valid_start_clip_idxs) == 0:
                idx = int(np.random.randint(0, len(self.uids)))
                continue

            start_clip_idx = int(np.random.choice(valid_start_clip_idxs))
            clip_idxs = [start_clip_idx + i for i in range(num_clips)]
            prompt = [all_captions[str(clip_idx)] for clip_idx in clip_idxs]
            break
        else:
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(np.random.choice(self.valid_idxs))

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        clip_length = int(round(self.CLIP_SECONDS * fps))
        start_frame_idx = int(round((start_clip_idx - 1) * self.CLIP_SECONDS * fps)) - self.CLIP_OVERLAP_FRAMES * (start_clip_idx - 1)
        total_clip_length = clip_length * num_clips - self.CLIP_OVERLAP_FRAMES * (num_clips - 1)

        depths, confs = None, None  # no depth and conf for InternalDataset

        # Load cameras
        if self.opt.load_da3_cam:
            # Sample frames on DA3's native 16fps timeline, then map back to source video frames
            full_16fps_src_frame_idxs = self._build_16fps_frame_indices(num_frames, float(fps), target_fps=self.DA3_FPS)
            if full_16fps_src_frame_idxs.shape[0] == 0:
                raise ValueError(f"Empty 16fps timeline for uid={uid}")

            clip_length_16fps = int(round(self.CLIP_SECONDS * self.DA3_FPS))
            da3_clip_start_idx = int(round((start_clip_idx - 1) * self.CLIP_SECONDS * self.DA3_FPS)) - self.CLIP_OVERLAP_FRAMES * (start_clip_idx - 1)
            da3_total_clip_length = clip_length_16fps * num_clips - self.CLIP_OVERLAP_FRAMES * (num_clips - 1)
            input_frame_idxs_16fps = self._frame_sample(
                full_16fps_src_frame_idxs.shape[0],
                start_frame_idx=da3_clip_start_idx,
                end_frame_idx=da3_clip_start_idx + da3_total_clip_length,
            )
            input_frame_idxs = full_16fps_src_frame_idxs[np.asarray(input_frame_idxs_16fps, dtype=np.int64)].tolist()

            extrinsics_chunks, intrinsics_chunks = [], []
            depth_chunks, conf_chunks = [], []
            da3_root = video_path.replace("video", "da3").replace(".mp4", "")
            for i, clip_idx in enumerate(clip_idxs):
                da3_dir = da3_root + f"/{clip_idx:02d}"
                _extrinsics = np.load(da3_dir + "/extrinsics.npy")
                _intrinsics = np.load(da3_dir + "/intrinsics.npy")
                trim_start = self.CLIP_OVERLAP_FRAMES if i > 0 else 0
                extrinsics_chunks.append(_extrinsics[trim_start:, ...])
                intrinsics_chunks.append(_intrinsics[trim_start:, ...])
                if self.opt.load_depth:
                    _depth = np.load(da3_dir + "/depth.npy")
                    depth_chunks.append(_depth[trim_start:, ...])
                if self.opt.load_conf:
                    _conf = np.load(da3_dir + "/conf.npy")
                    conf_chunks.append(_conf[trim_start:, ...])

            extrinsics_all = np.concatenate(extrinsics_chunks, axis=0)
            intrinsics_all = np.concatenate(intrinsics_chunks, axis=0)
            local_input_frame_idxs = np.asarray(input_frame_idxs_16fps, dtype=np.int64) - da3_clip_start_idx
            local_input_frame_idxs = np.clip(local_input_frame_idxs, 0, extrinsics_all.shape[0] - 1)

            W2C, intrinsics = extrinsics_all[local_input_frame_idxs, :, :], intrinsics_all[local_input_frame_idxs, :]
            W2C_ = torch.eye(4).unsqueeze(0).repeat(W2C.shape[0], 1, 1)
            W2C_[:, :3, :4] = torch.from_numpy(W2C).float()
            C2W = inverse_c2w(W2C_)  # (F, 4, 4); already in metric scale
            if W > H:  # landscape
                intrinsics[:, 0, 0] /= 504  # `504`: hard-coded
                intrinsics[:, 1, 1] /= 280  # `280`: hard-coded
                intrinsics[:, 0, 2] /= 504
                intrinsics[:, 1, 2] /= 280
            else:  # portrait
                intrinsics[:, 0, 0] /= 280  # `280`: hard-coded
                intrinsics[:, 1, 1] /= 504  # `504`: hard-coded
                intrinsics[:, 0, 2] /= 280
                intrinsics[:, 1, 2] /= 504
            fxfycxcy = intrinsics_to_fxfycxcy(torch.from_numpy(intrinsics).float()[None, ...])[0]  # (F, 4)

            if self.opt.load_depth:
                depths_all = np.concatenate(depth_chunks, axis=0)
                depths = torch.from_numpy(depths_all[local_input_frame_idxs, ...]).float()  # (F, H, W)
            if self.opt.load_conf:
                confs_all = np.concatenate(conf_chunks, axis=0)
                confs = torch.from_numpy(confs_all[local_input_frame_idxs, ...]).float()  # (F, H, W)

        else:
            # Keep the original VIPE loading logic when DA3 is disabled
            input_frame_idxs = self._frame_sample(
                num_frames,
                start_frame_idx=start_frame_idx,
                end_frame_idx=start_frame_idx + total_clip_length,
            )

            vipe_path = video_path.replace("video", "vipe").replace(".mp4", ".npz")
            vipe_data = np.load(vipe_path, allow_pickle=True)
            C2W, fxfycxcy = vipe_data["pose"], vipe_data["intrinsics"]
            assert C2W.shape[0] == fxfycxcy.shape[0] == num_frames
            C2W = torch.from_numpy(C2W).float()[input_frame_idxs, ...]  # (F, 4, 4)
            fxfycxcy = torch.from_numpy(fxfycxcy).float()[input_frame_idxs, ...]  # (F, 3, 3)
            fxfycxcy[:, 0] /= W
            fxfycxcy[:, 1] /= H
            fxfycxcy[:, 2] /= W
            fxfycxcy[:, 3] /= H

        if self.opt.load_image:
            # Load video
            images = {
                idx: tvT.ToTensor()(vr[idx].asnumpy())
                for idx in input_frame_idxs
            }
            images = torch.stack([images[idx] for idx in input_frame_idxs]).float()  # (F, 3, H, W)

            # Data augmentation
            images, depths, confs, fxfycxcy = self._data_augment(images, depths, confs, fxfycxcy)
        else:
            images = None

        # Camera normalization
        C2W = self._camera_normalize(C2W)

        # (Optional) Normalize XYZ
        scaling_factor = 1.
        if self.opt.normalize_xyz and depths is not None:
            _xyz = unproject_depth(depths[None, ...], C2W[None, ...], fxfycxcy[None, ...])[0]  # (F, 3, H, W)
            _xyz_norm = _xyz.norm(dim=1).mean().item()
            scaling_factor = 1. / (_xyz_norm + 1e-6)
            depths = depths * scaling_factor
            C2W[:, :3, 3] = C2W[:, :3, 3] * scaling_factor

        return_dict = {
            "uid": uid,            # str
            "prompt": prompt,      # List[str], len == num_clips
            "C2W": C2W,            # (F, 4, 4)
            "fxfycxcy": fxfycxcy,  # (F, 4)
        }
        if images is not None:
            return_dict["image"] = images  # (F, 3, H, W) in [0, 1]
        if depths is not None:
            return_dict["depth"] = depths  # (F, H, W)
        if confs is not None:
            return_dict["conf"] = confs  # (F, H, W)
        return return_dict

    @staticmethod
    def _build_16fps_frame_indices(num_frames: int, src_fps: float, target_fps: float = 16.) -> np.ndarray:
        # Keep exactly the same frame-index construction as resources/da3_internal.py
        if num_frames <= 0:
            return np.array([], dtype=np.int64)
        if src_fps <= 0:
            return np.arange(num_frames, dtype=np.int64)
        duration = num_frames / src_fps
        target_ts = np.arange(0., duration, 1. / target_fps, dtype=np.float32)
        frame_indices = np.floor(target_ts * src_fps).astype(np.int64)
        frame_indices = np.clip(frame_indices, 0, num_frames - 1)
        return frame_indices
