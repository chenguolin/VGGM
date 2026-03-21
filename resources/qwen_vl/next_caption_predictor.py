"""
Next-caption predictor -- importable module.

Quick start:
    from resources.qwen_vl.next_caption_predictor import NextCaptionPredictor

    predictor = NextCaptionPredictor(model_size="8B")

    # Single-step: predict next clip
    caption = predictor.predict("A person walks into a kitchen and opens the fridge.")

    # With frames
    caption = predictor.predict(
        "A drone flies over a mountain valley.",
        images=["frame_01.jpg", "frame_05.jpg"],
    )

    # With video
    caption = predictor.predict(
        "Cars race around a track under heavy rain.",
        video="clip.mp4",
    )

    # Multi-step autoregressive: predict clips 1..5 from clip 0
    predictions = predictor.predict_sequence(
        captions=["clip 0 caption", "clip 1 caption", ...],  # GT captions for all clips
        images_per_clip=[["f0_a.jpg", "f0_b.jpg"], None, ...],  # GT frames per clip (optional)
        num_steps=5,
    )
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info


ROOT = os.environ.get("ROOT", "/apdcephfs_sgfd/share_303967936/cglin")
HF_CACHE = os.path.join(ROOT, ".cache/huggingface/hub")

MODEL_PATHS = {
    "2B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-2B-Instruct"),
    "4B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-4B-Instruct"),
    "8B": os.path.join(HF_CACHE, "models--Qwen--Qwen3-VL-8B-Instruct"),
}

SYSTEM_PROMPT = (
    "You are a video understanding and prediction assistant. "
    "Given a description (caption) of the current video clip -- and optionally "
    "some visual frames or the video itself -- your task is to predict what will "
    "most likely happen in the NEXT video clip. "
    "Output ONLY the predicted caption for the next clip. "
    "The caption should be a single, concise paragraph (1-3 sentences) "
    "describing the visual content, actions, and scene of the next clip. "
    "Do NOT repeat the current caption; predict what comes next."
)


class NextCaptionPredictor:
    """Wraps Qwen3-VL for next-clip caption prediction."""

    def __init__(
        self,
        model_size: str = "8B",
        device: str = "cuda",
        system_prompt: Optional[str] = None,
    ):
        self.model_size = model_size
        self.device = device
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        model_path = self._resolve_model_path(model_size)
        print(f"Loading Qwen3-VL-{model_size} from {model_path} ...")

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
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

    def _build_user_content(
        self,
        caption: str,
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        sample_fps: int = 2,
        max_frames: int = 64,
    ) -> list[dict]:
        content = []
        has_visual = False

        if video is not None:
            content.append({
                "video": video,
                "max_frames": max_frames,
                "sample_fps": sample_fps,
                "total_pixels": 20480 * 32 * 32,
                "min_pixels": 64 * 32 * 32,
            })
            has_visual = True
        elif images is not None and len(images) > 0:
            for img_path in images:
                content.append({
                    "type": "image",
                    "image": img_path,
                    "min_pixels": 64 * 28 * 28,
                    "max_pixels": 512 * 28 * 28,
                })
            has_visual = True

        if has_visual:
            text = (
                f"The caption of this current video clip is:\n\"{caption}\"\n\n"
                "Based on the visual content shown above and the caption, "
                "predict and write the caption for the NEXT video clip "
                "(what will happen next). Output ONLY the predicted caption."
            )
        else:
            text = (
                f"The caption of the current video clip is:\n\"{caption}\"\n\n"
                "Based on this caption, predict and write the caption for the "
                "NEXT video clip (what will most likely happen next). "
                "Output ONLY the predicted caption."
            )

        content.append({"type": "text", "text": text})
        return content

    @torch.no_grad()
    def _generate(self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
    ) -> str:
        """Shared generation logic for both single and multi-step prediction."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True,
        )

        # Only pass `video_kwargs` when there are actual video inputs;
        # otherwise the empty lists (e.g. fps=[]) cause validation errors
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
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )

        if enable_thinking and "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        return output_text.strip()

    @torch.no_grad()
    def predict(
        self,
        caption: str,
        images: Optional[list[str]] = None,
        video: Optional[str] = None,
        max_new_tokens: int = 256,
        sample_fps: int = 2,
        max_frames: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
    ) -> str:
        """
        Predict the next clip's caption (single step).

        Args:
            caption: Caption of the current clip.
            images: Optional list of frame image paths.
            video: Optional video file path.
            max_new_tokens: Max generation length.
            sample_fps: Video sampling FPS.
            max_frames: Max frames from video.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            enable_thinking: Enable Qwen3 reasoning mode.

        Returns:
            Predicted caption for the next clip.
        """
        user_content = self._build_user_content(
            caption, images, video, sample_fps, max_frames,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        return self._generate(messages, max_new_tokens, temperature, top_p, enable_thinking)

    @torch.no_grad()
    def predict_sequence(
        self,
        captions: list[str],
        images_per_clip: Optional[list[Optional[list[str]]]] = None,
        videos_per_clip: Optional[list[Optional[str]]] = None,
        num_steps: Optional[int] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        step_callback=None,
    ) -> list[str]:
        """
        Autoregressive multi-step prediction.

        At each step, calls `predict()` with:
          - caption: the *predicted* caption from the previous step (or GT for step 0)
          - images/video: GT frames of the current clip (clip `step`)

        No conversation history is accumulated -- each step is an independent call.

        Args:
            captions: GT captions for all clips. Only captions[0] is used as the seed;
                      the rest are for evaluation outside this method.
            images_per_clip: Optional list of image path lists, one per clip.
                             images_per_clip[step] provides GT visual context for clip `step`.
            videos_per_clip: Optional list of video paths, one per clip.
            num_steps: Number of clips to predict. Default: len(captions) - 1.
            max_new_tokens: Max tokens per generation step.
            temperature: Sampling temperature.
            top_p: Nucleus sampling p.
            enable_thinking: Enable Qwen3 reasoning mode.
            step_callback: Optional callable(step, predicted_caption) called after each step.

        Returns:
            List of predicted captions for clips 1..num_steps.
        """
        if num_steps is None:
            num_steps = len(captions) - 1
        assert num_steps >= 1, "Need at least 1 step to predict"

        predictions = []
        current_caption = captions[0]

        for step in range(num_steps):
            # GT visual context for the current clip
            clip_images = None
            clip_video = None
            if images_per_clip is not None and step < len(images_per_clip):
                clip_images = images_per_clip[step]
            if videos_per_clip is not None and step < len(videos_per_clip):
                clip_video = videos_per_clip[step]

            # Single-turn prediction: predicted caption + GT frames -> next caption
            predicted = self.predict(
                caption=current_caption,
                images=clip_images,
                video=clip_video,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
            predictions.append(predicted)

            if step_callback is not None:
                step_callback(step, predicted)

            # Feed predicted caption to the next step
            current_caption = predicted

        return predictions

    def predict_batch(
        self,
        items: list[dict],
        **kwargs,
    ) -> list[str]:
        """
        Predict captions for a list of items.

        Each item: {"caption": str, "images": [...] or None, "video": str or None}
        Returns list of predicted captions.
        """
        results = []
        for i, item in enumerate(items):
            print(f"[{i+1}/{len(items)}] Predicting ...")
            pred = self.predict(
                caption=item["caption"],
                images=item.get("images"),
                video=item.get("video"),
                **kwargs,
            )
            results.append(pred)
        return results
