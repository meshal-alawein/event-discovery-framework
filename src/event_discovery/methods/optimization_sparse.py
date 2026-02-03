"""
Method 3: Pure Optimization (No Hierarchy)
Direct sparse selection without hierarchical filtering.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict
from dataclasses import dataclass

from ..core.video_processor import VideoWindow, VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for pure optimization method."""

    weight_novelty: float = 0.4
    weight_interaction: float = 0.3
    weight_uncertainty: float = 0.3

    top_k: int = 10
    diversity_weight: float = 0.5

    def normalize_weights(self):
        """Ensure weights sum to 1."""
        total = self.weight_novelty + self.weight_interaction + self.weight_uncertainty
        if total > 0:
            self.weight_novelty /= total
            self.weight_interaction /= total
            self.weight_uncertainty /= total


class PureOptimizationMethod:
    """
    Pure optimization without hierarchical filtering.

    Key difference from Hierarchical Energy:
    - Computes expensive features for ALL windows
    - No multi-scale pruning
    - Direct sparse selection via submodular optimization
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.config.normalize_weights()
        self.processor = VideoProcessor()

    def process_video(self, video_path: str) -> List[VideoWindow]:
        """Main pipeline: extract features -> score -> optimize."""
        windows = self.processor.chunk_video(video_path)
        logger.info("Chunked video into %d windows", len(windows))

        logger.info("Computing features for all windows (no filtering)...")
        features = self.extract_all_features(windows)

        scores = self.compute_scores(features)
        logger.info("Computed scores for %d windows", len(windows))

        selected = self.sparse_select(windows, scores)
        logger.info("Selected top-%d events via optimization", len(selected))

        return selected

    def extract_all_features(
        self, windows: List[VideoWindow]
    ) -> List[Dict[str, float]]:
        """Extract all features at maximum fidelity."""
        features = []

        for i, window in enumerate(windows):
            if i % 100 == 0:
                logger.debug("  Processing window %d/%d", i, len(windows))

            feat = {
                "novelty": self._compute_novelty(window),
                "interaction": self._compute_interaction(window),
                "uncertainty": self._compute_uncertainty(window),
            }
            features.append(feat)

        # Normalize features across the batch
        for key in ["novelty", "interaction", "uncertainty"]:
            values = np.array([f[key] for f in features])
            mean_val = np.mean(values)
            std_val = np.std(values)

            for feat in features:
                feat[key] = (
                    (feat[key] - mean_val) / (std_val + 1e-6)
                    if std_val > 1e-9
                    else 0.0
                )

        return features

    def _compute_novelty(self, window: VideoWindow) -> float:
        """Novelty: variance in pixel values as proxy for unusualness."""
        variances = [float(np.var(frame)) for frame in window.frames]
        return float(np.mean(variances))

    def _compute_interaction(self, window: VideoWindow) -> float:
        """Interaction density: edge detection variance as proxy."""
        edge_counts = []
        for frame in window.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_counts.append(np.sum(edges > 0))

        return float(np.std(edge_counts))

    def _compute_uncertainty(self, window: VideoWindow) -> float:
        """Model uncertainty: pixel value entropy as proxy."""
        entropies = []
        for frame in window.frames:
            hist, _ = np.histogram(frame.flatten(), bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-9)
            entropy = -np.sum(hist * np.log(hist + 1e-9))
            entropies.append(entropy)

        return float(np.mean(entropies))

    def compute_scores(self, features: List[Dict[str, float]]) -> np.ndarray:
        """S(W_i) = w_1*novelty + w_2*interaction + w_3*uncertainty"""
        scores = []
        for feat in features:
            score = (
                self.config.weight_novelty * feat["novelty"]
                + self.config.weight_interaction * feat["interaction"]
                + self.config.weight_uncertainty * feat["uncertainty"]
            )
            scores.append(score)
        return np.array(scores)

    def sparse_select(
        self, windows: List[VideoWindow], scores: np.ndarray
    ) -> List[VideoWindow]:
        """Greedy sparse selection with diversity."""
        selected_indices = []
        remaining = list(range(len(windows)))

        for _ in range(min(self.config.top_k, len(windows))):
            if not remaining:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                score = scores[idx]

                if selected_indices:
                    max_sim = max(
                        self._temporal_similarity(windows[idx], windows[sel_idx])
                        for sel_idx in selected_indices
                    )
                    score -= self.config.diversity_weight * max_sim

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [windows[i] for i in selected_indices]

    def _temporal_similarity(
        self, w1: VideoWindow, w2: VideoWindow
    ) -> float:
        """Temporal similarity penalty: exp(-|t1 - t2| / sigma)"""
        time_diff = abs(w1.start_time - w2.start_time)
        sigma = 10.0
        return float(np.exp(-time_diff / sigma))
