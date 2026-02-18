from typing import *
from torch import Tensor

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
        try:
            return self._try_getitem(idx)
        except Exception as e:
            # with open("dataload_error.log", "a") as f:
            #     f.write(f"Error processing idx={idx}, uid={self.uids[idx]}: {str(e)}\n")
            if idx in self.valid_idxs:
                self.valid_idxs.remove(idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalDataset!")
            return self.__getitem__(np.random.choice(self.valid_idxs))

    def _try_getitem(self, idx: int) -> Dict[str, Any]:
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

        # Video info
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        clip_length = int(round(self.CLIP_SECONDS * fps))

        # Camera info
        full_16fps_src_frame_idxs = None
        clip_length_16fps = None
        da3_root = None
        if self.opt.load_da3_cam:
            # Sample frames on DA3's native 16fps timeline, then map back to source video frames
            full_16fps_src_frame_idxs = self._build_16fps_frame_indices(num_frames, float(fps), target_fps=self.DA3_FPS)
            if full_16fps_src_frame_idxs.shape[0] == 0:
                raise ValueError(f"Empty 16fps timeline for uid={uid}")
            clip_length_16fps = int(round(self.CLIP_SECONDS * self.DA3_FPS))
            da3_root = video_path.replace("video", "da3").replace(".mp4", "")
        else:
            vipe_path = video_path.replace("video", "vipe").replace(".mp4", ".npz")
            vipe_data = np.load(vipe_path, allow_pickle=True)
            vipe_pose, vipe_intrinsics = vipe_data["pose"], vipe_data["intrinsics"]
            assert vipe_pose.shape[0] == vipe_intrinsics.shape[0] == num_frames

        # Load data for each video clip and aggregate into lists
        images_list, C2W_list, fxfycxcy_list = [], [], []
        depths_list: Optional[List[Tensor]] = [] if self.opt.load_depth else None
        confs_list: Optional[List[Tensor]] = [] if self.opt.load_conf else None
        frames_per_clip: List[int] = []

        for clip_idx in clip_idxs:
            start_frame_idx = int(round((clip_idx - 1) * self.CLIP_SECONDS * fps)) - self.CLIP_OVERLAP_FRAMES * (clip_idx - 1)
            depth_clip, conf_clip = None, None

            if self.opt.load_da3_cam:
                da3_clip_start_idx = int(round((clip_idx - 1) * self.CLIP_SECONDS * self.DA3_FPS)) - self.CLIP_OVERLAP_FRAMES * (clip_idx - 1)
                input_frame_idxs_16fps = self._frame_sample(
                    full_16fps_src_frame_idxs.shape[0],
                    start_frame_idx=da3_clip_start_idx,
                    end_frame_idx=da3_clip_start_idx + clip_length_16fps,
                    clip_idx=clip_idx,
                )
                input_frame_idxs = full_16fps_src_frame_idxs[np.asarray(input_frame_idxs_16fps, dtype=np.int64)].tolist()

                da3_dir = da3_root + f"/{clip_idx:02d}"
                extrinsics = np.load(da3_dir + "/extrinsics.npy")
                intrinsics = np.load(da3_dir + "/intrinsics.npy")
                local_input_frame_idxs = np.asarray(input_frame_idxs_16fps, dtype=np.int64) - da3_clip_start_idx
                local_input_frame_idxs = np.clip(local_input_frame_idxs, 0, extrinsics.shape[0] - 1)

                W2C = extrinsics[local_input_frame_idxs, :, :]
                W2C_ = torch.eye(4).unsqueeze(0).repeat(W2C.shape[0], 1, 1)
                W2C_[:, :3, :4] = torch.from_numpy(W2C).float()
                C2W = inverse_c2w(W2C_)  # (F, 4, 4)

                intrinsics = intrinsics[local_input_frame_idxs, ...].copy()
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
                    depth_clip = torch.from_numpy(np.load(da3_dir + "/depth.npy")[local_input_frame_idxs, ...]).float()  # (F, H, W)
                if self.opt.load_conf:
                    conf_clip = torch.from_numpy(np.load(da3_dir + "/conf.npy")[local_input_frame_idxs, ...]).float()  # (F, H, W)
            else:
                input_frame_idxs = self._frame_sample(
                    num_frames,
                    start_frame_idx=start_frame_idx,
                    end_frame_idx=start_frame_idx + clip_length,
                    clip_idx=clip_idx,
                )
                C2W = torch.from_numpy(vipe_pose).float()[input_frame_idxs, ...]  # (F, 4, 4)
                fxfycxcy = torch.from_numpy(vipe_intrinsics).float()[input_frame_idxs, ...]  # (F, 4)
                fxfycxcy[:, 0] /= W
                fxfycxcy[:, 1] /= H
                fxfycxcy[:, 2] /= W
                fxfycxcy[:, 3] /= H

            frames_per_clip.append(C2W.shape[0])
            C2W_list.append(C2W)
            fxfycxcy_list.append(fxfycxcy)

            images = {frame_idx: tvT.ToTensor()(vr[frame_idx].asnumpy()) for frame_idx in input_frame_idxs}
            images_list.append(torch.stack([images[frame_idx] for frame_idx in input_frame_idxs]).float())  # (F, 3, H, W)
            if self.opt.load_depth and depth_clip is not None:
                depths_list.append(depth_clip)
            if self.opt.load_conf and conf_clip is not None:
                confs_list.append(conf_clip)

        # Data augmentation on the entire concatenated clip sequence, then split back into clips
        images_all = torch.cat(images_list, dim=0)
        C2W_all = torch.cat(C2W_list, dim=0)
        fxfycxcy_all = torch.cat(fxfycxcy_list, dim=0)
        depths_all = torch.cat(depths_list, dim=0) if depths_list is not None and len(depths_list) > 0 else None
        confs_all = torch.cat(confs_list, dim=0) if confs_list is not None and len(confs_list) > 0 else None
        images_all, depths_all, confs_all, fxfycxcy_all = self._data_augment(images_all, depths_all, confs_all, fxfycxcy_all)

        images_list = list(torch.split(images_all, frames_per_clip, dim=0))
        C2W_list = list(torch.split(C2W_all, frames_per_clip, dim=0))
        fxfycxcy_list = list(torch.split(fxfycxcy_all, frames_per_clip, dim=0))
        if depths_all is not None and depths_list is not None:
            depths_list = list(torch.split(depths_all, frames_per_clip, dim=0))
        if confs_all is not None and confs_list is not None:
            confs_list = list(torch.split(confs_all, frames_per_clip, dim=0))

        # (Optional) Normalize XYZ
        scaling_factor = 1.
        if self.opt.normalize_xyz and depths_all is not None:
            _xyz = unproject_depth(depths_all[None, ...], C2W_all[None, ...], fxfycxcy_all[None, ...])[0]  # (F, 3, H, W)
            _xyz_norm = _xyz.norm(dim=1).mean().item()
            scaling_factor = 1. / (_xyz_norm + 1e-6)
            depths_all = depths_all * scaling_factor
            C2W_all[:, :3, 3] = C2W_all[:, :3, 3] * scaling_factor
            C2W_list = list(torch.split(C2W_all, frames_per_clip, dim=0))
            depths_list = list(torch.split(depths_all, frames_per_clip, dim=0))

        return_dict = {
            "uid": uid,                   # str
            "prompt": prompt,             # List[str] (len == num_clips)
            "C2W": C2W_list,              # List[(F, 4, 4)]
            "fxfycxcy": fxfycxcy_list,    # List[(F, 4)]
        }
        if self.opt.load_image:
            return_dict["image"] = images_list  # List[(F, 3, H, W)] in [0, 1]
        if depths_list is not None:
            return_dict["depth"] = depths_list  # List[(F, H, W)]
        if confs_list is not None:
            return_dict["conf"] = confs_list    # List[(F, H, W)]

        if self.opt.num_clips == 1:
            for key in ["C2W", "fxfycxcy"]:
                return_dict[key] = return_dict[key][0]  # (F, 4, 4) and (F, 4)
            if self.opt.load_image:
                return_dict["image"] = return_dict["image"][0]  # (F, 3, H, W)
            if depths_list is not None:
                return_dict["depth"] = return_dict["depth"][0]  # (F, H, W)
            if confs_list is not None:
                return_dict["conf"] = return_dict["conf"][0]    # (F, H, W)

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

    @staticmethod
    def _w2c3x4_to_homo4x4(w2c: np.ndarray) -> np.ndarray:
        w2c = np.asarray(w2c)
        w2c4 = np.broadcast_to(np.eye(4, dtype=np.float32), (w2c.shape[0], 4, 4)).copy()
        w2c4[:, :3, :4] = w2c.astype(np.float32)
        return w2c4

    @classmethod
    def _align_chunk_w2c_to_prev_chunk(
        cls,
        curr_chunk_w2c: np.ndarray,
        prev_overlap_w2c: np.ndarray,
        curr_overlap_w2c: np.ndarray,
    ) -> np.ndarray:
        # Align current chunk poses into previous chunk's world coordinate frame
        if curr_chunk_w2c.shape[0] == 0 or prev_overlap_w2c.shape[0] == 0 or curr_overlap_w2c.shape[0] == 0:
            return curr_chunk_w2c

        overlap = min(prev_overlap_w2c.shape[0], curr_overlap_w2c.shape[0])
        prev_overlap4 = cls._w2c3x4_to_homo4x4(prev_overlap_w2c[-overlap:])
        curr_overlap4 = cls._w2c3x4_to_homo4x4(curr_overlap_w2c[:overlap])

        # Use the overlap boundary pair to estimate world transform between chunks
        curr_to_prev = np.linalg.inv(curr_overlap4[-1]) @ prev_overlap4[-1]
        curr4 = cls._w2c3x4_to_homo4x4(curr_chunk_w2c)
        curr4_aligned = curr4 @ curr_to_prev[None, ...]
        return curr4_aligned[:, :3, :4]
