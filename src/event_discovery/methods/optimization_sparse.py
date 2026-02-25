"""
Method 3: Pure Optimization (No Hierarchy)
Direct sparse selection without hierarchical filtering.
"""

import logging
from dataclasses import dataclass

import numpy as np

from ..core.base import BaseEventDetector
from ..core.features import (
    compute_edge_density_variance,
    compute_pixel_entropy,
    compute_pixel_variance,
    normalize_features_batch,
)
from ..core.video_processor import VideoWindow

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for pure optimization method."""

    weight_novelty: float = 0.4
    weight_interaction: float = 0.3
    weight_uncertainty: float = 0.3

    top_k: int = 10
    diversity_weight: float = 0.5
    similarity_sigma: float = 10.0

    # Edge detection thresholds
    canny_low: int = 100
    canny_high: int = 200

    def normalize_weights(self):
        """Ensure weights sum to 1."""
        total = self.weight_novelty + self.weight_interaction + self.weight_uncertainty
        if total > 0:
            self.weight_novelty /= total
            self.weight_interaction /= total
            self.weight_uncertainty /= total


class PureOptimizationMethod(BaseEventDetector):
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
        super().__init__(
            top_k=self.config.top_k,
            diversity_weight=self.config.diversity_weight,
            sigma=self.config.similarity_sigma,
        )

    def _score_windows(self, windows: list[VideoWindow]) -> np.ndarray:
        """Compute scores for all windows at full fidelity."""
        features = self._extract_all_features(windows)
        return self._compute_scores(features)

    def _extract_all_features(self, windows: list[VideoWindow]) -> list[dict[str, float]]:
        """Extract all features at maximum fidelity."""
        raw_features = []

        for i, window in enumerate(windows):
            if i % 100 == 0:
                logger.debug("  Processing window %d/%d", i, len(windows))

            feat = {
                "novelty": compute_pixel_variance(window.frames),
                "interaction": compute_edge_density_variance(
                    window.frames, self.config.canny_low, self.config.canny_high
                ),
                "uncertainty": compute_pixel_entropy(window.frames),
            }
            raw_features.append(feat)

        return normalize_features_batch(raw_features)

    def _compute_scores(self, features: list[dict[str, float]]) -> np.ndarray:
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
