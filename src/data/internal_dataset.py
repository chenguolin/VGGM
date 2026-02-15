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
    DA3_FPS = 16.  # hard-coded

    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "internal", training)

        if opt.load_da3_cam:
            uids = os.listdir(f"{self.root}/da3")
        else:
            uids = os.listdir(f"{self.root}/valid_captions")
        indices = np.random.RandomState(seed=42).permutation(len(uids))
        if training:
            self.uids = [uids[i].strip(".json") for i in indices[:int(0.95 * len(uids))]]
        else:
            self.uids = [uids[i].strip(".json") for i in indices[int(0.95 * len(uids)):]]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        with open(f"{self.root}/valid_captions/{uid}.json", "r", encoding="utf-8") as f:
            all_captions = json.load(f)  # Dict[str, str]: clip_idx -> long caption
        dataset_source = "Internal"

        # Load prompt
        clip_idx = int(np.random.choice(list(all_captions.keys())))
        prompt = all_captions[str(clip_idx)]

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]
        # `5`: hard-coded for 5s-clip; `12`: hard-coded for clip-overlap
        start_frame_idx = int(round((clip_idx - 1) * 5 * fps)) - 12 * (clip_idx - 1)
        input_frame_idxs = self._frame_sample(
            num_frames,
            start_frame_idx=start_frame_idx,
            end_frame_idx=start_frame_idx + int(round(5 * fps)),
        )

        depths, confs = None, None  # no depth and conf for InternalDataset

        # Load cameras
        if self.opt.load_da3_cam:
            da3_dir = video_path.replace("video", "da3").replace(".mp4", "") + f"/{clip_idx:02d}"
            extrinsics_all = np.load(da3_dir + "/extrinsics.npy")
            intrinsics_all = np.load(da3_dir + "/intrinsics.npy")

            # DA3 preprocessing is done on a 16fps timeline (resources/da3_internal.py)
            da3_clip_start_idx = int(round((clip_idx - 1) * 5 * self.DA3_FPS)) - 12 * (clip_idx - 1)
            input_frame_idxs_16fps = np.floor(np.asarray(input_frame_idxs, dtype=np.float32) / float(fps) * self.DA3_FPS).astype(np.int64)
            local_input_frame_idxs = input_frame_idxs_16fps - da3_clip_start_idx
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
                depths = torch.from_numpy(np.load(da3_dir + "/depth.npy")[local_input_frame_idxs, ...]).float()  # (F, H, W)
            if self.opt.load_conf:
                confs = torch.from_numpy(np.load(da3_dir + "/conf.npy")[local_input_frame_idxs, ...]).float()  # (F, H, W)

        else:
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

            assert not self.opt.load_depth

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
            "prompt": prompt,      # str
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
