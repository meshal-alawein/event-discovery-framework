"""
Method 5: Dense VLM (Baseline)
Apply Vision-Language Model to all windows.
"""

import base64
import io
import logging
from typing import Optional

import numpy as np

from ..core.base import BaseEventDetector
from ..core.video_processor import VideoWindow

logger = logging.getLogger(__name__)


class VLMScoringError(Exception):
    """Raised when VLM scoring fails for a window."""


class DenseVLMMethod(BaseEventDetector):
    """
    Dense VLM baseline: Apply VLM to every window.

    This is the "oracle" method - maximum information per window
    but prohibitively expensive for long videos.

    Requires: OpenAI API key or local VLM model
    """

    def __init__(
        self,
        top_k: int = 10,
        model: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None,
        use_local: bool = False,
    ):
        super().__init__(top_k=top_k, diversity_weight=0.0)
        self.model = model
        self.api_key = api_key
        self.use_local = use_local
        self.client = None

        if not use_local and api_key is None:
            raise ValueError("Must provide api_key or set use_local=True")

        if not use_local:
            try:
                import openai

                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install event-discovery[vlm]"
                ) from None

    def _score_windows(self, windows: list[VideoWindow]) -> np.ndarray:
        """Score all windows using VLM."""
        scores = []
        for i, window in enumerate(windows):
            if i % 10 == 0:
                logger.info("  Scoring window %d/%d", i, len(windows))
            try:
                score = self._score_window(window)
                scores.append(score)
            except VLMScoringError as e:
                logger.warning("Failed to score window %d: %s", i, e)
                scores.append(0.0)
        return np.array(scores)

    def _select(self, windows: list[VideoWindow], scores: np.ndarray) -> list[VideoWindow]:
        """Select top-k by score (no diversity penalty for VLM method)."""
        top_indices = np.argsort(scores)[-self.top_k :][::-1]
        return [windows[i] for i in top_indices]

    def _score_window(self, window: VideoWindow) -> float:
        """Score a single window. Returns 0-1."""
        if self.use_local:
            return self._score_with_local_model(window)
        else:
            return self._score_with_openai(window)

    def _score_with_openai(self, window: VideoWindow) -> float:
        """Score window using OpenAI GPT-4V."""

        key_frames = self._sample_key_frames(window, n_frames=4)
        encoded_frames = [self._encode_image(frame) for frame in key_frames]

        prompt = (
            "Analyze these sequential frames from a driving video. "
            "Rate the importance of this scene on a scale of 0-10, where: "
            "0-2: Normal, routine driving; 3-5: Minor notable events; "
            "6-8: Significant events (near-misses); 9-10: Critical events "
            "(violations, accidents). Respond with ONLY a single number 0-10."
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in encoded_frames
                ],
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=10, temperature=0.0
            )
            score_text = response.choices[0].message.content.strip()
            return float(score_text) / 10.0
        except (ValueError, AttributeError, IndexError) as e:
            raise VLMScoringError(f"Failed to parse VLM response: {e}") from e
        except Exception as e:
            raise VLMScoringError(f"API request failed: {e}") from e

    def _score_with_local_model(self, window: VideoWindow) -> float:
        """Score window using local VLM. Not yet implemented."""
        raise NotImplementedError("Local VLM scoring not yet implemented")

    def _sample_key_frames(self, window: VideoWindow, n_frames: int = 4) -> list[np.ndarray]:
        """Sample key frames from window for VLM."""
        num_frames = len(window.frames)
        if num_frames <= n_frames:
            return list(window.frames)
        indices = np.linspace(0, num_frames - 1, n_frames, dtype=int)
        return [window.frames[i] for i in indices]

    def _encode_image(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG for API."""
        from PIL import Image

        frame_rgb = frame[:, :, ::-1]
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")


def estimate_cost(video_duration_minutes: float, window_size_seconds: float = 2.0):
    """Estimate API cost for dense VLM processing."""
    num_windows = int((video_duration_minutes * 60) / window_size_seconds)
    images_per_window = 4
    cost_per_image = 0.01

    total_cost = num_windows * images_per_window * cost_per_image

    return {
        "video_duration_minutes": video_duration_minutes,
        "num_windows": num_windows,
        "total_images": num_windows * images_per_window,
        "estimated_cost_usd": total_cost,
    }
