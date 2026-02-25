"""
Method 1: Hierarchical Energy-Based Event Discovery
Physics-inspired approach with multi-scale filtering.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from ..core.base import BaseEventDetector
from ..core.features import (
    compute_color_histogram,
    compute_edge_density_variance,
    compute_pixel_variance,
    greedy_diverse_select,
    normalize_features_batch,
)
from ..core.video_processor import VideoWindow

logger = logging.getLogger(__name__)


@dataclass
class EnergyConfig:
    """Configuration for energy functional."""

    # Feature weights (alpha_k)
    weight_motion: float = 0.3
    weight_interaction: float = 0.3
    weight_scene_change: float = 0.2
    weight_uncertainty: float = 0.2

    # Adaptive threshold multipliers (sigma_l) per level
    sigma_multipliers: list[float] = field(
        default_factory=lambda: [2.0, 1.5, 1.0]
    )

    # Sparse selection
    top_k: int = 10
    diversity_weight: float = 0.5
    similarity_sigma: float = 10.0

    # Edge detection thresholds
    canny_low: int = 100
    canny_high: int = 200

    # Histogram bins
    histogram_bins: int = 32

    def normalize_weights(self):
        """Ensure weights sum to 1."""
        total = (
            self.weight_motion
            + self.weight_interaction
            + self.weight_scene_change
            + self.weight_uncertainty
        )
        if total > 0:
            self.weight_motion /= total
            self.weight_interaction /= total
            self.weight_scene_change /= total
            self.weight_uncertainty /= total

    @property
    def num_levels(self) -> int:
        return len(self.sigma_multipliers)


class HierarchicalEnergyMethod(BaseEventDetector):
    """
    Hierarchical energy-based event discovery.

    Key components:
    1. Event energy functional E(W) = sum alpha_k * phi_k(W)
    2. Multi-scale thresholding (renormalization)
    3. Sparse optimization with diversity
    """

    def __init__(self, config: EnergyConfig = None):
        self.config = config or EnergyConfig()
        self.config.normalize_weights()
        super().__init__(
            top_k=self.config.top_k,
            diversity_weight=self.config.diversity_weight,
            sigma=self.config.similarity_sigma,
        )

    def process_video(self, video_path: str) -> list[VideoWindow]:
        """
        Override base pipeline to insert hierarchical filtering.
        """
        windows = self.processor.chunk_video(video_path)
        logger.info("Chunked video into %d windows", len(windows))

        candidates = self._hierarchical_filter(windows)
        logger.info("Filtered to %d candidates", len(candidates))

        selected = self._select_from_candidates(candidates)
        logger.info("Selected %d events", len(selected))

        return selected

    def _score_windows(self, windows: list[VideoWindow]) -> np.ndarray:
        """Score windows using full energy computation at max fidelity."""
        features = self._extract_features(windows, level=self.config.num_levels - 1)
        return self._compute_energy(features)

    def _hierarchical_filter(self, windows: list[VideoWindow]) -> list[VideoWindow]:
        """
        Multi-scale energy-based filtering.

        for l = 0 to L:
            compute E_l(W_i) at fidelity level l
            filter: C_{l+1} = {W : E_l(W) > tau_l}
        """
        candidates = windows

        for level in range(self.config.num_levels):
            if len(candidates) == 0:
                break

            logger.info("  Level %d: %d candidates", level, len(candidates))

            features = self._extract_features(candidates, level=level)
            energies = self._compute_energy(features)
            tau = self._adaptive_threshold(energies, level)

            logger.info(
                "    Threshold: %.3f (mean=%.3f, std=%.3f)",
                tau, np.mean(energies), np.std(energies),
            )

            mask = energies > tau
            candidates = [w for w, m in zip(candidates, mask) if m]

        return candidates

    def _extract_features(
        self, windows: list[VideoWindow], level: int = 0
    ) -> list[dict[str, float]]:
        """
        Extract features at given fidelity level.

        Level 0: Cheap features (motion, scene change)
        Level 1: Medium features (+ interaction)
        Level 2: Expensive features (+ uncertainty)
        """
        raw_features = []

        for window in windows:
            feat = {
                "motion": self._compute_motion_energy(window),
                "scene_change": self._compute_scene_change(window),
                "interaction": (
                    compute_edge_density_variance(
                        window.frames, self.config.canny_low, self.config.canny_high
                    )
                    if level >= 1
                    else 0.0
                ),
                "uncertainty": (
                    compute_pixel_variance(window.frames) if level >= 2 else 0.0
                ),
            }
            raw_features.append(feat)

        return normalize_features_batch(raw_features)

    def _compute_motion_energy(self, window: VideoWindow) -> float:
        """Motion energy: ||v_dot||^2 + ||v_ddot||^2"""
        flow = self.processor.compute_optical_flow(window)
        velocity = np.sqrt(flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2)
        v_mean = np.mean(velocity, axis=(1, 2))
        acceleration = np.diff(v_mean)
        return float(np.sum(v_mean**2) + np.sum(acceleration**2))

    def _compute_scene_change(self, window: VideoWindow) -> float:
        """Scene change: L2 distance between start and end histograms."""
        hist_start = compute_color_histogram(window.frames[0], bins=self.config.histogram_bins)
        hist_end = compute_color_histogram(window.frames[-1], bins=self.config.histogram_bins)
        return float(np.linalg.norm(hist_end - hist_start))

    def _compute_energy(self, features: list[dict[str, float]]) -> np.ndarray:
        """E(W_i) = sum alpha_k * phi_k(W_i)"""
        energies = []
        for feat in features:
            energy = (
                self.config.weight_motion * feat.get("motion", 0)
                + self.config.weight_interaction * feat.get("interaction", 0)
                + self.config.weight_scene_change * feat.get("scene_change", 0)
                + self.config.weight_uncertainty * feat.get("uncertainty", 0)
            )
            energies.append(energy)
        return np.array(energies)

    def _adaptive_threshold(self, energies: np.ndarray, level: int) -> float:
        """tau_l = mean(E) + sigma_l * std(E)"""
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        sigma_mult = self.config.sigma_multipliers[level]
        return float(mean_energy + sigma_mult * std_energy)

    def _select_from_candidates(self, candidates: list[VideoWindow]) -> list[VideoWindow]:
        """Score filtered candidates and apply diverse selection."""
        if len(candidates) <= self.top_k:
            return candidates

        features = self._extract_features(candidates, level=self.config.num_levels - 1)
        scores = self._compute_energy(features)

        return greedy_diverse_select(
            candidates=candidates,
            scores=scores,
            top_k=self.top_k,
            diversity_weight=self.diversity_weight,
            sigma=self.sigma,
        )
