from typing import *

import os
import hashlib
import pickle
import numpy as np
import json
from decord import VideoReader, cpu
import torch

from src.options import Options
from src.data.base_dataset import BaseDataset


class _SkipSample(Exception):
    """Raised inside `_getitem_once` to signal that the current sample should be skipped."""
    def __init__(self, idx: int):
        self.idx = idx


class InternalActionDataset(BaseDataset):
    """Dataset for the new action-grounded video data format (v2).

    Each sample has structured captions per segment:
      - `caption_abs`: absolute scene description
      - `caption_delta`: change from previous segment
      - `end_state`: state after the action completes
      - `action_label`: short action tag
    Plus per-video fields:
      - `global_caption`: one-sentence video summary
      - `control_agent`: detailed appearance description of the main agent
    """

    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "internal", training)

        # Fast path: load pre-built UID list from local /tmp cache
        cache_path = self._cache_key(opt, self.root, training)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.uids = cached["uids"]
            self.caption_data = cached["caption_data"]
            self.valid_idxs = list(range(len(self.uids)))
            return

        # Load JSONL data
        data_path = os.path.join(self.root, opt.action_data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Build caption data: uid -> caption_result dict
        self.caption_data = {item["filename"]: item["caption_result"] for item in data}
        uids = list(self.caption_data.keys())

        # Train/val split (95/5)
        indices = np.random.RandomState(seed=42).permutation(len(uids))
        if training:
            self.uids = [uids[i] for i in indices[:int(.95 * len(uids))]]
        else:
            self.uids = [uids[i] for i in indices[int(.95 * len(uids)):]]

        # Filter UIDs: one `listdir` + set lookup instead of many `os.path.exists` calls
        existing_videos = set(os.listdir(os.path.join(self.root, "video")))
        self.uids = [uid for uid in self.uids if f"{uid}.mp4" in existing_videos]
        if self.opt.input_plucker:
            existing_vipe = set(os.listdir(os.path.join(self.root, "vipe")))
            self.uids = [uid for uid in self.uids if f"{uid}.npz" in existing_vipe]
        self.valid_idxs = list(range(len(self.uids)))

        # Save to /tmp cache so future runs on this node skip the slow remote listdir
        cached = {"uids": self.uids, "caption_data": self.caption_data}
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cached, f, protocol=4)
        except Exception:
            pass  # non-fatal

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Iterative retry loop -- avoids RecursionError when many samples fail consecutively
        tried = set()
        current_idx = idx
        while True:
            tried.add(current_idx)
            try:
                return self._getitem_once(current_idx)
            except _SkipSample as e:
                bad_idx = e.idx
                if bad_idx in self.valid_idxs:
                    self.valid_idxs.remove(bad_idx)
                if len(self.valid_idxs) == 0:
                    raise ValueError("No valid data in InternalActionDataset!")
                current_idx = int(np.random.choice(self.valid_idxs))

    def _getitem_once(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        caption_result = self.caption_data[uid]
        segments = caption_result["segments"]

        # Extract structured captions per segment
        all_captions_abs = [seg["caption"]["caption_abs"] for seg in segments]
        all_action_labels = [seg["action_label"] for seg in segments]
        all_caption_deltas = [seg["caption"]["caption_delta"] for seg in segments]
        all_end_states = [seg["caption"]["end_state"] for seg in segments]

        # Dynamic `num_clips`: randomly pick a start segment, then greedily extend
        # until duration falls in [min_duration, max_duration]
        target_frames_per_clip = self.opt.num_input_frames - 1
        min_duration = target_frames_per_clip / 32.  # TODO: make configurable
        max_duration = target_frames_per_clip / 16.  # TODO: make configurable
        n_segs = len(segments)

        # Build valid (start, num_clips) pairs
        valid_pairs = []  # (start_idx, num_clips)
        for si in range(n_segs):
            cum_dur = 0.
            for length in range(1, n_segs - si + 1):
                cum_dur += float(segments[si + length - 1]["end_time"]) - float(segments[si + length - 1]["start_time"])
                if cum_dur > max_duration * length:
                    break  # adding more segments won't help
                if min_duration * length <= cum_dur <= max_duration * length:
                    valid_pairs.append((si, length))

        # Filter out pairs with too many clips or any segment too short
        max_clips = self.opt.num_clips * 3  # TODO: make configurable
        min_seg_duration = 1.  # TODO: make configurable

        def _pair_is_valid(si, nc):
            if nc > max_clips:
                return False
            for k in range(nc):
                seg = segments[si + k]
                if float(seg["end_time"]) - float(seg["start_time"]) < min_seg_duration:
                    return False
            return True

        valid_pairs = [(si, nc) for si, nc in valid_pairs if _pair_is_valid(si, nc)]
        if len(valid_pairs) == 0:
            raise _SkipSample(idx)
        start_clip_idx, num_clips = valid_pairs[np.random.randint(len(valid_pairs))]
        clip_idxs = [start_clip_idx + i for i in range(num_clips)]

        # Prompts: use `caption_abs` as the DiT prompt
        prompt = [all_captions_abs[ci] for ci in clip_idxs]
        action_labels = [all_action_labels[ci] for ci in clip_idxs]
        caption_deltas = [all_caption_deltas[ci] for ci in clip_idxs]
        end_states = [all_end_states[ci] for ci in clip_idxs]

        # Sample frames
        video_path = os.path.join(self.root, "video", f"{uid}.mp4")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames, fps, (H, W) = len(vr), vr.get_avg_fps(), vr[0].shape[:2]

        # Re-create `vr` with lower decode resolution to save CPU memory
        new_H, new_W = self.opt.input_res
        scale = max(new_H / H, new_W / W)
        if scale < 1.:
            del vr
            vr = VideoReader(str(video_path), ctx=cpu(0), width=round(W * scale), height=round(H * scale))

        # Calculate total frames based on `opt.num_clips` (keep total frame count fixed
        # across samples; dynamic `num_clips` only controls how many segments are selected)
        total_frames_clips = self.opt.num_clips
        if not self.training:
            total_frames_clips = self.opt.num_clips_test
        total_frames = (self.opt.num_input_frames - 1) * total_frames_clips + 1
        if self.opt.is_causal:  # make sure video latents can be divided by the causal chunk size
            total_frames_latent = 1 + (total_frames - 1) // self.opt.compression_ratio[0]
            total_frames_latent = int(np.ceil(total_frames_latent / self.opt.chunk_size) * self.opt.chunk_size)
            total_frames = 1 + (total_frames_latent - 1) * self.opt.compression_ratio[0]

        # Compute frame indices via per-segment proportional allocation
        selected_segments = [segments[ci] for ci in clip_idxs]
        seg_frame_ranges = []  # (seg_start_frame, seg_end_frame) for each segment
        for seg in selected_segments:
            seg_start = int(round(float(seg["start_time"]) * fps))
            seg_end = min(int(round(float(seg["end_time"]) * fps)), num_frames)
            if seg_end < seg_start + min_seg_duration * fps:
                raise _SkipSample(idx)
            seg_frame_ranges.append((seg_start, seg_end))
        seg_num_frames = [(end - start) for start, end in seg_frame_ranges]
        total_seg_frames = sum(seg_num_frames)

        # Allocate `total_frames` proportionally to each segment's real frame count
        raw_alloc = [(total_frames * n / total_seg_frames) for n in seg_num_frames]
        num_frames_per_clip = [int(round(a)) for a in raw_alloc]

        # Fix rounding residual: adjust the largest clip
        residual = total_frames - sum(num_frames_per_clip)
        if residual != 0:
            largest_idx = int(np.argmax(num_frames_per_clip))
            num_frames_per_clip[largest_idx] += residual

        # Sample frames independently within each segment
        input_frame_idxs = []
        for (seg_start, seg_end), target_f in zip(seg_frame_ranges, num_frames_per_clip):
            seg_all_frames = np.arange(seg_start, seg_end, dtype=int)
            sampled = seg_all_frames[np.linspace(0, len(seg_all_frames) - 1, target_f, dtype=int)]
            input_frame_idxs.extend(sampled.tolist())

        # Relative physical timestamps (seconds): first frame is 0, preserving real-world
        # inter-frame time intervals from the original video
        timestamps = torch.tensor([fi / fps for fi in input_frame_idxs], dtype=torch.float32)  # (F,)
        timestamps = timestamps - timestamps[0]  # shift so first frame = 0

        depths, confs = None, None  # no depth and conf for this dataset

        if self.opt.input_plucker:
            # Load cameras (in metric scale)
            vipe_path = video_path.replace("video", "vipe").replace(".mp4", ".npz")
            vipe_data = np.load(vipe_path, allow_pickle=True)
            C2W, fxfycxcy = vipe_data["pose"], vipe_data["intrinsics"]
            if (C2W.shape[0] != fxfycxcy.shape[0]) or (C2W.shape[0] != num_frames):
                raise _SkipSample(idx)
            C2W = torch.from_numpy(C2W).float()[input_frame_idxs, ...]  # (F, 4, 4)
            fxfycxcy = torch.from_numpy(fxfycxcy).float()[input_frame_idxs, ...]  # (F, 3, 3)
            fxfycxcy[:, 0] /= W
            fxfycxcy[:, 1] /= H
            fxfycxcy[:, 2] /= W
            fxfycxcy[:, 3] /= H
        else:
            C2W, fxfycxcy = None, None

        if self.opt.load_image:
            # Load video
            frames = vr.get_batch(input_frame_idxs).asnumpy()  # (F, H, W, C) uint8
            del vr
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.  # (F, 3, H, W)
            del frames

            # Data augmentation
            images, depths, confs, fxfycxcy = self._data_augment(images, depths, confs, fxfycxcy)
        else:
            del vr
            images = None

        if C2W is not None:
            # Camera normalization
            C2W = self._camera_normalize(C2W)

        # Split into clips
        assert sum(num_frames_per_clip) == len(input_frame_idxs) and len(num_frames_per_clip) == len(prompt)
        timestamps = torch.split(timestamps, num_frames_per_clip, dim=0)  # List of (F_i,)
        if C2W is not None:
            C2W = torch.split(C2W, num_frames_per_clip, dim=0)  # List of (F_i, 4, 4)
        if fxfycxcy is not None:
            fxfycxcy = torch.split(fxfycxcy, num_frames_per_clip, dim=0)  # List of (F_i, 4)
        if images is not None:
            images = torch.split(images, num_frames_per_clip, dim=0)  # List of (F_i, 3, H, W)

        return_dict = {
            "uid": uid,                                      # str
            "prompt": prompt,                                # List[str] (caption_abs)
            "action_labels": action_labels,                  # List[str]
            "caption_deltas": caption_deltas,                # List[str]
            "end_states": end_states,                        # List[str]
            "global_caption": caption_result["global_caption"],  # str
            "control_agent": caption_result["control_agent"],    # str
            "timestamps": timestamps,                        # List[Tensor(F_i,)] in seconds
        }
        if self.opt.load_image:
            return_dict["image"] = images  # List[(F, 3, H, W)] in [0, 1]
        if self.opt.input_plucker:
            return_dict["C2W"] = C2W  # List[(F, 4, 4)]
            return_dict["fxfycxcy"] = fxfycxcy  # List[(F, 4)]

        # Cumulative time offsets for each clip at 16 fps: [(start_sec, end_sec), ...]
        cum = 0
        frame_ranges = []
        for n in num_frames_per_clip:
            frame_ranges.append((cum / 16., (cum + n) / 16.))  # TODO: make `16` configurable
            cum += n
        return_dict["frame_ranges"] = frame_ranges  # List[Tuple[float, float]]

        # Unpack the single clip (based on actual num_clips, not opt.num_clips)
        if len(prompt) == 1:
            for key in return_dict:
                if key not in ("uid", "global_caption", "control_agent"):
                    return_dict[key] = return_dict[key][0]

        return return_dict

    @staticmethod
    def _cache_key(opt: Options, root: str, training: bool) -> str:
        """Cache key captures all factors that affect `self.uids`."""
        factors = (
            root,
            training,
            opt.action_data_path,
            opt.input_plucker,
        )
        key = hashlib.md5(str(factors).encode()).hexdigest()[:12]
        split = "train" if training else "val"
        return f"/tmp/internal_action_dataset_uids_{split}_{key}.pkl"
