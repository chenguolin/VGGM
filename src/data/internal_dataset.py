from typing import *

import os
import numpy as np
import json
from decord import VideoReader, cpu
import torch

from src.options import Options
from src.data.base_dataset import BaseDataset


class InternalDataset(BaseDataset):
    def __init__(self, opt: Options, training: bool = True):
        super().__init__(opt, "internal", training)

        # 4. Version: action
        if self.opt.version_action:
            with open(f"{self.root}/videos_action_caption.jsonl", "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            ## Fix typo keys (e.g. `end_end`, `end_end_time`, `end_speed` -> `end_time`)
            for item in data:
                for seg in item["caption_result"]["segments"]:
                    if "end_time" not in seg:
                        for k in list(seg.keys()):
                            if k not in ("segment_id", "start_time", "action_label", "caption"):
                                seg["end_time"] = seg.pop(k)
                                break
            ## `caption_data`: `uid` -> {`whole_caption`, `segments`: [{`segment_id`, `start_time`, `end_time`, `action_label`, `caption`}, ...]}
            self.caption_data = {item["filename"]: item["caption_result"] for item in data}
            if self.opt.load_global_caption:
                self.global_caption_data = {item["filename"]: item["caption_result"]["whole_caption"] for item in data}
            uids = list(self.caption_data.keys())

        # 3. Version: 2sdiff
        elif self.opt.version_2sdiff:
            with open(f"{self.root}/valid_captions_2sdiff.jsonl", "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            self.caption_data = {item["raw_id"]: [clip["caption"] for clip in item["long_caption_lst"]] for item in data}
            if self.opt.load_global_caption:
                self.global_caption_data = {item["raw_id"]: item["org_long_caption"] for item in data}
            uids = list(self.caption_data.keys())

        # 2. Version: 2s35w
        elif self.opt.version_2s35w:
            with open(f"{self.root}/valid_captions_2s35w.jsonl", "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            self.caption_data = {item["raw_id"]: item["long_caption"] for item in data}
            if self.opt.load_global_caption:
                self.global_caption_data = {item["raw_id"]: item["org_long_caption"] for item in data}
            uids = list(self.caption_data.keys())

        # 1. Version: 5s40w
        else:
            uids = os.listdir(f"{self.root}/valid_captions")

        indices = np.random.RandomState(seed=42).permutation(len(uids))
        if self.opt.version_action or self.opt.version_2s35w or self.opt.version_2sdiff:
            if training:
                self.uids = [uids[i] for i in indices[:int(0.95 * len(uids))]]
            else:
                self.uids = [uids[i] for i in indices[int(0.95 * len(uids)):]]
        else:
            if training:
                self.uids = [uids[i].strip(".json") for i in indices[:int(0.95 * len(uids))]]
            else:
                self.uids = [uids[i].strip(".json") for i in indices[int(0.95 * len(uids)):]]

        # Filter UIDs: one `listdir` + set lookup instead of 300k `os.path.exists` calls
        existing_videos = set(os.listdir(os.path.join(self.root, "video")))
        self.uids = [uid for uid in self.uids if f"{uid}.mp4" in existing_videos]
        if self.opt.input_plucker:
            existing_vipe = set(os.listdir(os.path.join(self.root, "vipe")))
            self.uids = [uid for uid in self.uids if f"{uid}.npz" in existing_vipe]
        self.valid_idxs = list(range(len(self.uids)))

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid = self.uids[idx]
        if self.opt.version_action:
            caption_result = self.caption_data[uid]
            segments = caption_result["segments"]
            all_captions = [seg["caption"] for seg in segments]
            all_clip_idxs = list(range(len(segments)))  # 0-based
            clip_duration, clip_base, clip_overlap = None, 0, 0  # variable-length segments
        elif self.opt.version_2s35w or self.opt.version_2sdiff:
            all_captions = self.caption_data[uid]  # List[str]: 0-based clip captions
            all_clip_idxs = list(range(len(all_captions)))  # 0-based
            clip_duration, clip_base, clip_overlap = 2, 0, 0
        else:
            with open(f"{self.root}/valid_captions/{uid}.json", "r", encoding="utf-8") as f:
                all_captions = json.load(f)  # Dict[str, str]: clip_idx -> long caption
            all_clip_idxs = sorted([int(k) for k in all_captions.keys()])  # 1-based
            clip_duration, clip_base, clip_overlap = 5, 1, 12
        dataset_source = "Internal"

        all_clip_idx_set = set(all_clip_idxs)

        # Sample `num_clips` and `start_clip_idx`, and get multiple prompts accordingly
        if self.opt.version_action:
            ## Dynamic `num_clips` for `version_action`: randomly pick a start segment, then
            ## greedily extend until duration falls in [min_duration, max_duration].
            target_frames_per_clip = self.opt.num_input_frames - 1
            min_duration, max_duration = target_frames_per_clip / 32., target_frames_per_clip / 8.  # TODO: make `32` / `8` configurable
            n_segs = len(segments)
            ## Build valid (start, num_clips) pairs
            valid_pairs = []  # (start_idx, num_clips)
            for si in range(n_segs):
                cum_dur = 0.
                for length in range(1, n_segs - si + 1):
                    cum_dur += segments[si + length - 1]["end_time"] - segments[si + length - 1]["start_time"]
                    if cum_dur > max_duration * length:
                        break  # adding more segments won't help
                    if min_duration * length <= cum_dur <= max_duration * length:
                        valid_pairs.append((si, length))
            ## Filter out pairs with too many clips or any segment too short
            max_clips = self.opt.num_clips * 2  # TODO: make `2` configurable
            min_seg_duration = 1.  # TODO: make `1.` configurable
            def _pair_is_valid(si, nc):
                if nc > max_clips:
                    return False
                for k in range(nc):
                    seg = segments[si + k]
                    if seg["end_time"] - seg["start_time"] < min_seg_duration:
                        return False
                return True
            valid_pairs = [(si, nc) for si, nc in valid_pairs if _pair_is_valid(si, nc)]
            if len(valid_pairs) == 0:
                if idx in self.valid_idxs:
                    self.valid_idxs.remove(idx)
                    if len(self.valid_idxs) == 0:
                        raise ValueError("No valid data in InternalDataset!")
                return self.__getitem__(int(np.random.choice(self.valid_idxs)))
            start_clip_idx, num_clips = valid_pairs[np.random.randint(len(valid_pairs))]
            clip_idxs = [start_clip_idx + i for i in range(num_clips)]
            prompt = [all_captions[clip_idx] for clip_idx in clip_idxs]
        else:
            ## Fixed `num_clips` for other dataset versions
            max_num_clips = max(1, int(self.opt.num_clips))
            if self.training:
                num_clips = max_num_clips if not self.opt.random_num_clips else \
                    np.random.randint(1, max_num_clips + 1)
            else:
                num_clips = max_num_clips  # fixed to `max_num_clips` for evaluation

            valid_start_clip_idxs = [
                clip_idx for clip_idx in all_clip_idxs
                if all((clip_idx + i) in all_clip_idx_set for i in range(num_clips))
            ]
            if len(valid_start_clip_idxs) == 0:
                if idx in self.valid_idxs:
                    self.valid_idxs.remove(idx)
                    if len(self.valid_idxs) == 0:
                        raise ValueError("No valid data in InternalDataset!")
                return self.__getitem__(int(np.random.choice(self.valid_idxs)))

            if self.opt.version_2sdiff and 0 in valid_start_clip_idxs:
                ### For `version_2sdiff`, always start from clip 0 because clip 0 captions have a
                ### distinct introductory style vs. continuation style in later clips
                start_clip_idx = 0
            else:
                start_clip_idx = int(np.random.choice(valid_start_clip_idxs))
            clip_idxs = [start_clip_idx + i for i in range(num_clips)]  # consecutive clip indices
            if self.opt.version_2s35w or self.opt.version_2sdiff:
                prompt = [all_captions[clip_idx] for clip_idx in clip_idxs]
            else:
                prompt = [all_captions[str(clip_idx)] for clip_idx in clip_idxs]

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
        # Calculate total frames based on `num_clips`
        # For `version_action`, use `opt.num_clips` to keep total frame count fixed across
        # samples (dynamic `num_clips` only controls how many segments are selected)
        total_frames_clips = self.opt.num_clips if self.opt.version_action else num_clips
        if not self.training:
            total_frames_clips = self.opt.num_clips_test
        total_frames = (self.opt.num_input_frames - 1) * total_frames_clips + 1
        if self.opt.is_causal:  # make sure video latents can be divided by the causal chunk size
            total_frames_latent = 1 + (total_frames - 1) // self.opt.compression_ratio[0]
            total_frames_latent = int(np.ceil(total_frames_latent / self.opt.chunk_size) * self.opt.chunk_size)
            total_frames = 1 + (total_frames_latent - 1) * self.opt.compression_ratio[0]

        # Compute frame indices and per-clip frame counts
        if self.opt.version_action:
            selected_segments = [segments[ci] for ci in clip_idxs]
            ## Per-segment proportional frame allocation and independent uniform sampling
            seg_frame_ranges = []  # (seg_start_frame, seg_end_frame) for each segment
            for seg in selected_segments:
                seg_start = int(round(seg["start_time"] * fps))
                seg_end = min(int(round(seg["end_time"] * fps)), num_frames)
                seg_frame_ranges.append((seg_start, seg_end))
            seg_num_frames = [max(end - start, 1) for start, end in seg_frame_ranges]  # real frame count per segment
            total_seg_frames = sum(seg_num_frames)
            ## Allocate `total_frames` proportionally to each segment's real frame count
            raw_alloc = [total_frames * n / total_seg_frames for n in seg_num_frames]
            num_frames_per_clip = [max(1, int(round(a))) for a in raw_alloc]
            ## Fix rounding residual: adjust the largest clip
            residual = total_frames - sum(num_frames_per_clip)
            if residual != 0:
                largest_idx = int(np.argmax(num_frames_per_clip))
                num_frames_per_clip[largest_idx] += residual
            ## Sample frames independently within each segment
            input_frame_idxs = []
            for (seg_start, seg_end), target_f in zip(seg_frame_ranges, num_frames_per_clip):
                seg_all_frames = np.arange(seg_start, seg_end, dtype=int)
                sampled = seg_all_frames[np.linspace(0, len(seg_all_frames) - 1, target_f, dtype=int)]
                input_frame_idxs.extend(sampled.tolist())
        else:
            ## `clip_duration`: seconds per clip; `clip_overlap`: inter-clip overlap in frames; `clip_base`: index offset
            ci_first = clip_idxs[0] - clip_base
            start_frame_idx = int(round(ci_first * clip_duration * fps)) - clip_overlap * ci_first
            end_frame_idx = start_frame_idx + num_clips * int(round(clip_duration * fps))  # may exceed video length
            input_frame_idxs = self._frame_sample(
                num_frames,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                target_num_frames=total_frames,
            )

        depths, confs = None, None  # no depth and conf for InternalDataset

        if self.opt.input_plucker:
            # Load cameras (in metric scale)
            vipe_path = video_path.replace("video", "vipe").replace(".mp4", ".npz")
            vipe_data = np.load(vipe_path, allow_pickle=True)
            C2W, fxfycxcy = vipe_data["pose"], vipe_data["intrinsics"]
            if (C2W.shape[0] != fxfycxcy.shape[0]) or (C2W.shape[0] != num_frames):
                if idx in self.valid_idxs:
                    self.valid_idxs.remove(idx)
                    if len(self.valid_idxs) == 0:
                        raise ValueError("No valid data in InternalDataset!")
                return self.__getitem__(int(np.random.choice(self.valid_idxs)))
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
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (F, 3, H, W)
            del frames

            # Data augmentation
            images, depths, confs, fxfycxcy = self._data_augment(images, depths, confs, fxfycxcy)
        else:
            del vr
            images = None

        if C2W is not None:
            # Camera normalization
            C2W = self._camera_normalize(C2W)

        # Split into clips (`num_frames_per_clip` already computed for `version_action`)
        if not self.opt.version_action:
            num_frames_per_clip = []
            for ii, clip_idx in enumerate(clip_idxs):
                ci = clip_idx - clip_base
                if ii == 0:
                    clip_start_frame_idx = int(round(ci * clip_duration * fps)) - clip_overlap * ci
                else:  # to avoid overlapping frames between consecutive clips, we start the next clip from the end of the previous clip
                    clip_start_frame_idx = clip_end_frame_idx
                clip_end_frame_idx = clip_start_frame_idx + int(round(clip_duration * fps))  # may exceed video length
                num_frames_per_clip.append(len([idx for idx in input_frame_idxs if clip_start_frame_idx <= idx < clip_end_frame_idx]))
        assert sum(num_frames_per_clip) == len(input_frame_idxs) and len(num_frames_per_clip) == len(prompt)
        if C2W is not None:
            C2W = torch.split(C2W, num_frames_per_clip, dim=0)  # List of (F_i, 4, 4)
        if fxfycxcy is not None:
            fxfycxcy = torch.split(fxfycxcy, num_frames_per_clip, dim=0)  # List of (F_i, 4)
        if images is not None:
            images = torch.split(images, num_frames_per_clip, dim=0)  # List of (F_i, 3, H, W)

        return_dict = {
            "uid": uid,            # str
            "prompt": prompt,      # List[str]
        }
        if self.opt.load_image:
            return_dict["image"] = images  # List[(F, 3, H, W)] in [0, 1]
        if self.opt.input_plucker:
            return_dict["C2W"] = C2W  # List[(F, 4, 4)]
            return_dict["fxfycxcy"] = fxfycxcy  # List[(F, 4)]

        if self.opt.load_global_caption:
            return_dict["global_caption"] = self.global_caption_data[uid]  # str
        if self.opt.version_action:
            selected_segs = [segments[ci] for ci in clip_idxs]
            return_dict["action_labels"] = [seg["action_label"] for seg in selected_segs]  # List[str]

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
                if key != "uid":
                    return_dict[key] = return_dict[key][0]

        return return_dict
