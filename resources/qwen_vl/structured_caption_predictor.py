"""
Structured caption predictor for action-grounded video data.

Predicts per-segment structured captions (action_label, caption_abs,
caption_delta, end_state) plus per-video fields (global_caption,
control_agent) in an autoregressive manner.

User only provides video frames; all text fields are inferred by the VLM.

Quick start::

    from resources.qwen_vl.structured_caption_predictor import StructuredCaptionPredictor

    predictor = StructuredCaptionPredictor(model_size="8B")

    # Bootstrap: infer global context + first segment from frames
    seg0 = predictor.predict_initial(images=["frame_00.jpg", "frame_01.jpg"])
    # seg0 = {
    #     "global_caption": "...",
    #     "control_agent": "...",
    #     "segment": {"action_label": "...", "caption_abs": "...", "end_state": "..."},
    # }

    # Autoregressive: predict next segment given new frames + history
    seg1 = predictor.predict_next(
        images=["frame_10.jpg", "frame_11.jpg"],
        global_caption=seg0["global_caption"],
        control_agent=seg0["control_agent"],
        history=[seg0["segment"]],
    )
    # seg1 = {"action_label": "...", "caption_delta": "...", "end_state": "...", "estimated_frames": ...}

    # Full autoregressive sequence
    results = predictor.predict_sequence(
        images_per_segment=[["f0a.jpg", "f0b.jpg"], ["f1a.jpg", "f1b.jpg"], ...],
    )
"""

import json
import os
import re
from typing import Optional

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


ROOT = os.environ.get("ROOT", "/apdcephfs_sgfd/share_303967936/cglin")
HF_CACHE = os.path.join(ROOT, ".cache/huggingface/hub")

MODEL_PATHS = {
    # Qwen3-VL
    "2B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-2B-Instruct"),
    "4B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-4B-Instruct"),
    "8B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-8B-Instruct"),
    # Qwen3.5 (omni VL, no "-VL" suffix in HF name)
    "3.5-0.8B": os.path.join(HF_CACHE, "models--Qwen--Qwen3.5-0.8B"),
    "3.5-2B": os.path.join(HF_CACHE, "models--Qwen--Qwen3.5-2B"),
    "3.5-4B": os.path.join(HF_CACHE, "models--Qwen--Qwen3.5-4B"),
    "3.5-9B": os.path.join(HF_CACHE, "models--Qwen--Qwen3.5-9B"),
}

SYSTEM_PROMPT_INITIAL = """\
You are a video understanding assistant. Given frames from a video segment, \
produce a structured caption in JSON format.

You must output a JSON object with these fields:
- "global_caption": A 2-3 sentence summary of the overall video scene, \
setting, and narrative context.
- "control_agent": A detailed physical description of the main character or \
agent (appearance, clothing, distinguishing features). If no clear agent, \
describe the main subject.
- "segment": An object with:
  - "action_label": A short imperative phrase (5-10 words) describing the \
main action in this segment.
  - "caption_abs": A 2-4 sentence description of the scene, environment, \
and what happens in this segment.
  - "end_state": One sentence describing the visual state at the end of \
this segment.

Output ONLY valid JSON. No markdown fences, no extra text."""

SYSTEM_PROMPT_NEXT = """\
You are a video prediction assistant. Given the overall video context, \
what happened in previous segments (text + key frames), predict what will \
happen in the NEXT segment.

You must output a JSON object with these fields:
- "action_label": A short imperative phrase (5-10 words) predicting the \
main action in the next segment.
- "caption_delta": 1-2 sentences predicting what will CHANGE compared to \
the previous segment (new actions, scene changes, camera movement, etc.).
- "end_state": One sentence predicting the visual state at the end of \
the next segment.
- "estimated_frames": An integer estimating how many video frames (at 16fps) \
the next segment will last.

Output ONLY valid JSON. No markdown fences, no extra text."""

SYSTEM_PROMPT_JUDGE = """\
You are a video generation quality judge. Given frames from a video \
segment and the intended end state describing what the scene SHOULD \
look like when the action is complete, decide whether to stop, \
continue, or regenerate.

You must output a JSON object with exactly two fields, in this order:
- "reason": One sentence explaining your reasoning.
- "verdict": "stop" if the intended end state has been reached and \
the visual quality is acceptable, "continue" if the action is still \
in progress, "regenerate" if the content is wrong or quality is poor.

Output ONLY valid JSON. No markdown fences, no extra text."""


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from model output, tolerating markdown fences."""
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    return json.loads(text)


class StructuredCaptionPredictor:
    """Predicts structured per-segment captions using Qwen3-VL."""

    def __init__(
        self,
        model_size: str = "8B",
        device: str = "cuda",
        lora_path: Optional[str] = None,
        model_path: Optional[str] = None,
        max_pixels: int = 256 * 28 * 28,
    ):
        self.model_size = model_size
        self.device = device
        self.max_pixels = max_pixels  # ~200K -> ~450x450 -> ~196 tokens/image

        # `model_path` overrides the default HF cache path (for full FT checkpoints)
        base_path = model_path or self._resolve_model_path(model_size)
        print(f"Loading Qwen3-VL-{model_size} from {base_path} ...")

        self.processor = AutoProcessor.from_pretrained(
            base_path, trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        if lora_path is not None:
            from peft import PeftModel
            print(f"Loading LoRA adapter from {lora_path} ...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()
            print("LoRA adapter merged.")

        self.model.eval()
        print("Model loaded.")

    @staticmethod
    def _resolve_model_path(model_size: str) -> str:
        base = MODEL_PATHS[model_size]
        snapshot_dir = os.path.join(base, "snapshots")
        if os.path.isdir(snapshot_dir):
            snapshots = os.listdir(snapshot_dir)
            if snapshots:
                return os.path.join(snapshot_dir, snapshots[0])
        return base

    def _build_image_content(
        self,
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        sample_fps: int = 2,
        max_frames: int = 64,
    ) -> list[dict]:
        """Build the visual part of user content."""
        content = []
        if video is not None:
            content.append({
                "video": video,
                "max_frames": max_frames,
                "sample_fps": sample_fps,
                "total_pixels": 20480 * 32 * 32,
                "min_pixels": 64 * 32 * 32,
            })
        elif images is not None and len(images) > 0:
            for img_path in images:
                content.append({
                    "type": "image",
                    "image": img_path,
                    "min_pixels": 64 * 28 * 28,
                    "max_pixels": self.max_pixels,
                })
        return content

    @torch.no_grad()
    def _generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.,
        top_p: float = 0.9,
        enable_thinking: bool = False,
    ) -> str:
        """Run generation and return raw text output."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True,
        )

        processor_kwargs = {}
        if video_inputs is not None:
            processor_kwargs.update(video_kwargs)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            **processor_kwargs,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

        generated_ids = output_ids[0, inputs.input_ids.shape[1]:]
        output_text = self.processor.decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if enable_thinking and "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        return output_text.strip()

    @torch.no_grad()
    def predict_initial(
        self,
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.,
        enable_thinking: bool = False,
    ) -> dict:
        """Bootstrap: infer global_caption, control_agent, and first segment.

        Args:
            images: Frame image paths for the first segment.
            video: Video file path for the first segment.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature (0 = greedy).
            enable_thinking: Enable Qwen3 reasoning mode.

        Returns:
            Dict with keys: ``global_caption``, ``control_agent``, ``segment``
            (which has ``action_label``, ``caption_abs``, ``end_state``).
        """
        visual_content = self._build_image_content(images, video)
        user_content = visual_content + [{
            "type": "text",
            "text": (
                "These are frames from the first segment of a video. "
                "Analyze them and produce the structured JSON caption."
            ),
        }]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_INITIAL},
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(
            messages, max_new_tokens, temperature,
            enable_thinking=enable_thinking,
        )
        result = _parse_json(raw)

        # Validate expected keys
        for key in ("global_caption", "control_agent", "segment"):
            if key not in result:
                raise ValueError(
                    f"Missing key '{key}' in initial prediction. "
                    f"Raw output:\n{raw}"
                )
        seg = result["segment"]
        for key in ("action_label", "caption_abs", "end_state"):
            if key not in seg:
                raise ValueError(
                    f"Missing key 'segment.{key}' in initial prediction. "
                    f"Raw output:\n{raw}"
                )

        return result

    @torch.no_grad()
    def predict_next(
        self,
        global_caption: str = "",
        control_agent: str = "",
        history: Optional[list[dict]] = None,
        history_images: Optional[list[Optional[list[str]]]] = None,
        history_window: int = 0,
        max_new_tokens: int = 512,
        temperature: float = 0.,
        enable_thinking: bool = False,
    ) -> dict:
        """Predict the next segment from history context (text + frames).

        The VLM does NOT see the next segment's frames — it predicts what
        should happen next based solely on prior segments.

        Args:
            global_caption: Overall video summary (from bootstrap or user).
            control_agent: Main agent description (from bootstrap or user).
            history: List of previous segment dicts, each with
                ``action_label``, ``caption_abs``, ``end_state``, etc.
            history_images: List of image path lists for each history
                segment.  When provided, key frames from previous segments
                are included as visual context alongside their text
                descriptions.
            history_window: Max recent segments to include in full detail.
                When > 0, seg0 is kept as a sink, middle segments are
                compressed into an action-label chain, and only the last
                W segments keep full text + frames.  0 = no windowing.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature (0 = greedy).
            enable_thinking: Enable Qwen3 reasoning mode.

        Returns:
            Dict with keys: ``action_label``, ``caption_abs``,
            ``caption_delta``, ``end_state``, ``estimated_frames``.
        """
        if history is None:
            history = []
        if history_images is None:
            history_images = [None] * len(history)

        # Build interleaved visual + text context for history segments
        user_content: list[dict] = []

        # Text preamble
        user_content.append({
            "type": "text",
            "text": f"Video overview: {global_caption}\nMain character: {control_agent}",
        })

        # ── Partition history: sink + skipped + window ────────────
        n_hist = len(history)
        use_window = history_window > 0 and n_hist > history_window + 1

        if use_window:
            sink_idx = [0]
            window_start = n_hist - history_window
            skipped_idx = list(range(1, window_start))
            window_idx = list(range(window_start, n_hist))
        else:
            sink_idx = []
            skipped_idx = []
            window_idx = list(range(n_hist))

        def _seg_text(i, seg):
            """Format segment text with frame count."""
            if i == 0:
                scene = seg.get("caption_abs", "N/A")
            else:
                scene = seg.get("caption_delta", "") or seg.get("caption_abs", "N/A")
            est = seg.get("estimated_frames", "")
            frames_str = f" ({est} frames)" if est else ""
            desc = (
                f"Segment {i + 1}{frames_str}: "
                f"[{seg.get('action_label', 'N/A')}] {scene}"
            )
            end = seg.get("end_state", "")
            if end:
                desc += f" End state: {end}"
            return desc

        def _add_images(i):
            """Append image entries for segment *i*."""
            imgs = history_images[i] if i < len(history_images) else None
            if imgs:
                for img_path in imgs:
                    user_content.append({
                        "type": "image",
                        "image": img_path,
                        "min_pixels": 64 * 28 * 28,
                        "max_pixels": self.max_pixels,
                    })

        # ── Build history ─────────────────────────────────────────
        if history:
            user_content.append({"type": "text", "text": "Previous segments:"})

            # Sink (seg0): always full text + frames
            for i in sink_idx:
                _add_images(i)
                user_content.append({"type": "text", "text": _seg_text(i, history[i])})

            # Skipped: action-label chain summary
            if skipped_idx:
                labels = []
                for i in skipped_idx:
                    seg = history[i]
                    est = seg.get("estimated_frames", "")
                    f_str = f" ({est}f)" if est else ""
                    labels.append(f"{seg.get('action_label', '?')}{f_str}")
                user_content.append({
                    "type": "text",
                    "text": (
                        f"[Segments {skipped_idx[0] + 1}-{skipped_idx[-1] + 1} "
                        f"summary: " + " → ".join(labels) + "]"
                    ),
                })

            # Window: full text + frames
            for i in window_idx:
                _add_images(i)
                user_content.append({"type": "text", "text": _seg_text(i, history[i])})

        # Instruction
        user_content.append({
            "type": "text",
            "text": (
                "Based on the video context and previous segments above, "
                "predict what will happen in the NEXT segment. "
                "Output the structured JSON prediction."
            ),
        })

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NEXT},
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(
            messages, max_new_tokens, temperature,
            enable_thinking=enable_thinking,
        )
        result = _parse_json(raw)

        # Validate expected keys
        for key in ("action_label", "caption_delta", "end_state"):
            if key not in result:
                raise ValueError(
                    f"Missing key '{key}' in next-segment prediction. "
                    f"Raw output:\n{raw}"
                )

        return result

    @torch.no_grad()
    def judge(
        self,
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        intended_caption: Optional[dict] = None,
        control_agent: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.,
        enable_thinking: bool = False,
    ) -> dict:
        """Judge whether video frames match the intended structured caption.

        Args:
            images: Frame image paths for the segment being judged.
            video: Video file path for the segment being judged.
            intended_caption: Dict with ``action_label``, ``caption_abs``,
                ``end_state`` describing what SHOULD happen.
            control_agent: Main agent description.
            max_new_tokens: Max generation length.
            temperature: Sampling temperature (0 = greedy).
            enable_thinking: Enable Qwen3 reasoning mode.

        Returns:
            Dict with ``reason`` and ``verdict``.
        """
        import json as _json

        visual_content = self._build_image_content(images, video)

        caption_json = _json.dumps({
            "end_state": intended_caption.get("end_state", ""),
        }, ensure_ascii=False)

        user_content = visual_content + [{
            "type": "text",
            "text": (
                f"Intended end state:\n{caption_json}\n\n"
                "Evaluate whether this video segment has reached "
                "the intended end state."
            ),
        }]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(
            messages, max_new_tokens, temperature,
            enable_thinking=enable_thinking,
        )
        result = _parse_json(raw)
        return result

    @torch.no_grad()
    def predict_sequence(
        self,
        images_per_segment: Optional[list[Optional[list[str]]]] = None,
        videos_per_segment: Optional[list[Optional[str]]] = None,
        history_vision: str = "all",
        history_window: int = 0,
        initial_context: Optional[dict] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.,
        enable_thinking: bool = False,
        step_callback=None,
    ) -> dict:
        """Full autoregressive prediction over multiple segments.

        User provides only visual input per segment; all captions are inferred.

        Args:
            images_per_segment: List of image path lists, one per segment.
            videos_per_segment: List of video paths, one per segment.
            history_vision: How much visual history to include in context:
                - ``"none"``: text-only history, no history frames.
                - ``"last"``: only the previous segment's key frames.
                - ``"all"``: all history segments' key frames.
            initial_context: If provided, skip ``predict_initial`` and use
                this dict directly.  Expected keys:

                - ``"global_caption"`` (str): overall video summary.
                - ``"control_agent"`` (str): main agent description.
                - ``"segment"`` (dict): first segment with at least
                  ``action_label``, ``caption_abs``, ``end_state``.

                When ``None``, the first segment's visual input is sent to
                ``predict_initial`` to bootstrap these fields.
            max_new_tokens: Max tokens per generation step.
            temperature: Sampling temperature (0 = greedy).
            enable_thinking: Enable Qwen3 reasoning mode.
            step_callback: Optional callable(step_idx, segment_dict) called
                after each segment is predicted.

        Returns:
            Dict with ``global_caption``, ``control_agent``, and
            ``segments`` (list of per-segment dicts).
        """
        assert history_vision in ("none", "last", "all")

        n_segments = 0
        if images_per_segment is not None:
            n_segments = len(images_per_segment)
        elif videos_per_segment is not None:
            n_segments = len(videos_per_segment)
        assert n_segments >= 1, "Need at least 1 segment"

        def _get_images(i):
            if images_per_segment is not None and i < len(images_per_segment):
                return images_per_segment[i]
            return None

        def _get_video(i):
            if videos_per_segment is not None and i < len(videos_per_segment):
                return videos_per_segment[i]
            return None

        # Step 0: bootstrap or use user-provided context
        if initial_context is not None:
            global_caption = initial_context["global_caption"]
            control_agent = initial_context["control_agent"]
            segments = [initial_context["segment"]]
        else:
            initial = self.predict_initial(
                images=_get_images(0),
                video=_get_video(0),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
            global_caption = initial["global_caption"]
            control_agent = initial["control_agent"]
            segments = [initial["segment"]]

        if step_callback is not None:
            step_callback(0, segments[0])

        # Steps 1..N-1: autoregressive
        all_history_images: list[Optional[list[str]]] = [_get_images(0)]
        for i in range(1, n_segments):
            # Select visual history based on `history_vision` mode
            if history_vision == "none":
                hist_images = [None] * len(segments)
            elif history_vision == "last":
                hist_images = [None] * (len(segments) - 1) + [all_history_images[-1]]
            else:  # "all"
                hist_images = list(all_history_images)

            seg = self.predict_next(
                global_caption=global_caption,
                control_agent=control_agent,
                history=segments,
                history_images=hist_images,
                history_window=history_window,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
            segments.append(seg)
            all_history_images.append(_get_images(i))

            if step_callback is not None:
                step_callback(i, seg)

        return {
            "global_caption": global_caption,
            "control_agent": control_agent,
            "segments": segments,
        }
