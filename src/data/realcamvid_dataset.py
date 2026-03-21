from typing import *

import os
import numpy as np
from decord import VideoReader, cpu
import torch

from src.options import Options
from src.data.base_dataset import BaseDataset
from src.utils.geo_util import inverse_c2w, intrinsics_to_fxfycxcy, unproject_depth


class RealcamvidDataset(BaseDataset):
    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "realcamvid", training)

        if training:
            self.metadata = np.load(f"{self.root}/RealCam-Vid_train.npz", allow_pickle=True)["arr_0"]
        else:
            self.metadata = np.load(f"{self.root}/RealCam-Vid_test.npz", allow_pickle=True)["arr_0"]

        self.valid_idxs = list(range(len(self.metadata)))

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        metadata = self.metadata[idx]
        dataset_source = metadata["dataset_source"]  # "RealEstate10K", "DL3DV-10K", "MiraData9K"
        uid = metadata["video_path"].replace("/", "_").replace(".mp4", "")

        if self.opt.only_static_data:
            if "Mira" in uid:
                return self.__getitem__(np.random.choice(self.valid_idxs))

        # Load prompt
        # if np.random.rand() < 0.75:  # TODO: make it configurable
        #     prompt = metadata["long_caption"]
        # else:
        #     prompt = metadata["short_caption"]
        if self.opt.use_short_caption:
            prompt = metadata["short_caption"]
        else:
            prompt = metadata["long_caption"]

        # Sample frames
        video_path = os.path.join(self.root, metadata["video_path"])
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames = len(vr)
        # Re-create `vr` with lower decode resolution to save CPU memory
        H, W = vr[0].shape[:2]
        new_H, new_W = self.opt.input_res
        scale = max(new_H / H, new_W / W)
        if scale < 1.:
            del vr
            vr = VideoReader(str(video_path), ctx=cpu(0), width=round(W * scale), height=round(H * scale))
        input_frame_idxs = self._frame_sample(num_frames,
            pingpong_threshold=self.opt.pingpong_threshold if "Mira" not in uid else -1)  # not reverse videos for dynamic MiraData

        depths, confs = None, None

        # Load cameras
        if self.opt.load_da3_cam:
            da3_path = video_path.replace("RealCam-Vid", "RealCam-Vid-DA3").replace(".mp4", ".npz")
            da3_data = np.load(da3_path, allow_pickle=True)
            W2C, intrinsics = da3_data["extrinsics"][input_frame_idxs, ...], da3_data["intrinsics"][input_frame_idxs, ...]
            W2C_ = torch.eye(4).unsqueeze(0).repeat(W2C.shape[0], 1, 1)
            W2C_[:, :3, :4] = torch.from_numpy(W2C).float()
            C2W = inverse_c2w(W2C_)  # (F, 4, 4); already in metric scale
            intrinsics[:, 0, 0] /= 504  # `504`: hard-coded
            intrinsics[:, 1, 1] /= 280  # `280`: hard-coded
            intrinsics[:, 0, 2] /= 504
            intrinsics[:, 1, 2] /= 280
            fxfycxcy = intrinsics_to_fxfycxcy(torch.from_numpy(intrinsics).float()[None, ...])[0]  # (F, 4)

            if self.opt.load_depth or self.opt.normalize_xyz:
                depths = torch.from_numpy(da3_data["depth"][input_frame_idxs, ...]).float()  # (F, H, W)
            if self.opt.load_conf:
                confs = torch.from_numpy(da3_data["conf"][input_frame_idxs, ...]).float()  # (F, H, W)

        else:
            W2C = metadata["camera_extrinsics"]
            if num_frames != W2C.shape[0]:
                if idx in self.valid_idxs:
                    self.valid_idxs.remove(idx)
                    if len(self.valid_idxs) == 0:
                        raise ValueError("No valid data in RealcamvidDataset!")
                return self.__getitem__(np.random.choice(self.valid_idxs))

            C2W = inverse_c2w(torch.from_numpy(W2C).float())[input_frame_idxs, ...]  # (F, 4, 4)
            C2W[:, :3, 3] *= metadata["align_factor"]  # to metric scale
            fxfycxcy = torch.from_numpy(metadata["camera_intrinsics"]).float()[None, :].repeat(C2W.shape[0], 1)  # (F, 4)

            assert not self.opt.load_depth

        if self.opt.load_image:
            # Load video
            frames = vr.get_batch(input_frame_idxs).asnumpy()  # (F, H, W, C) uint8
            del vr
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (F, 3, H, W)
            del frames

            # Data augmentation
            images, depths, confs, fxfycxcy = self._data_augment(images, depths, confs, fxfycxcy)
        else:
            del vr
            images = None

        # Camera normalization
        C2W = self._camera_normalize(C2W)

        # (Optional) Normalize XYZ
        scaling_factor = 1.
        if self.opt.normalize_xyz:
            assert depths is not None  # need depth to compute the scaling factor
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
        if self.opt.load_image:
            return_dict["image"] = images  # (F, 3, H, W) in [0, 1]
        if self.opt.load_depth:
            return_dict["depth"] = depths  # (F, H, W)
        if self.opt.load_conf:
            return_dict["conf"] = confs  # (F, H, W)
        return return_dict
