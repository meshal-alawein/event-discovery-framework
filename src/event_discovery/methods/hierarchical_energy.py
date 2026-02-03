"""
Method 1: Hierarchical Energy-Based Event Discovery
Physics-inspired approach with multi-scale filtering.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict
from dataclasses import dataclass, field

from ..core.video_processor import VideoWindow, VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class EnergyConfig:
    """Configuration for energy functional."""

    # Feature weights (alpha_k)
    weight_motion: float = 0.3
    weight_interaction: float = 0.3
    weight_scene_change: float = 0.2
    weight_uncertainty: float = 0.2

    # Adaptive threshold multipliers (sigma_l)
    sigma_multipliers: List[float] = field(
        default_factory=lambda: [2.0, 1.5, 1.0]
    )

    # Sparse selection
    top_k: int = 10
    diversity_weight: float = 0.5

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


class HierarchicalEnergyMethod:
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
        self.processor = VideoProcessor()

    def process_video(self, video_path: str) -> List[VideoWindow]:
        """
        Main pipeline: hierarchical filtering + sparse selection.

        Args:
            video_path: Path to video file

        Returns:
            List of top-k detected event windows
        """
        windows = self.processor.chunk_video(video_path)
        logger.info("Chunked video into %d windows", len(windows))

        candidates = self.hierarchical_filter(windows)
        logger.info("Filtered to %d candidates", len(candidates))

        selected = self.sparse_select(candidates)
        logger.info("Selected top-%d events", len(selected))

        return selected

    def hierarchical_filter(
        self, windows: List[VideoWindow]
    ) -> List[VideoWindow]:
        """
        Multi-scale energy-based filtering.

        Algorithm:
        for l = 0 to L:
            compute E_l(W_i) at fidelity level l
            filter: C_{l+1} = {W : E_l(W) > tau_l}
        """
        candidates = windows

        for level in range(self.config.num_levels):
            if len(candidates) == 0:
                break

            logger.info(
                "  Level %d: %d candidates", level, len(candidates)
            )

            features = self.extract_features(candidates, level=level)
            energies = self.compute_energy(features)
            tau = self.adaptive_threshold(energies, level)

            logger.info(
                "    Threshold: %.3f (mean=%.3f, std=%.3f)",
                tau,
                np.mean(energies),
                np.std(energies),
            )

            mask = energies > tau
            candidates = [w for w, m in zip(candidates, mask) if m]

        return candidates

    def extract_features(
        self, windows: List[VideoWindow], level: int = 0
    ) -> List[Dict[str, float]]:
        """
        Extract features at given fidelity level.

        Level 0: Cheap features (motion, scene change)
        Level 1: Medium features (+ interaction)
        Level 2: Expensive features (+ uncertainty)
        """
        raw_features = []

        for window in windows:
            feat = {}
            feat["motion"] = self._compute_motion_energy(window)
            feat["scene_change"] = self._compute_scene_change(window)

            if level >= 1:
                feat["interaction"] = self._compute_interaction(window)
            else:
                feat["interaction"] = 0.0

            if level >= 2:
                feat["uncertainty"] = self._compute_uncertainty(window)
            else:
                feat["uncertainty"] = 0.0

            raw_features.append(feat)

        # Normalize features across all windows (not incrementally)
        features = self._normalize_features(raw_features)
        return features

    def _normalize_features(
        self, raw_features: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Z-score normalize features across the full batch."""
        if not raw_features:
            return raw_features

        keys = raw_features[0].keys()
        normalized = [{} for _ in raw_features]

        for key in keys:
            values = np.array([f[key] for f in raw_features])
            mean_val = np.mean(values)
            std_val = np.std(values)
            for i, val in enumerate(values):
                normalized[i][key] = (
                    (val - mean_val) / (std_val + 1e-6) if std_val > 1e-9 else 0.0
                )

        return normalized

    def _compute_motion_energy(self, window: VideoWindow) -> float:
        """
        Motion energy: ||v_dot||^2 + ||v_ddot||^2

        Physics: Kinetic deviation from steady-state
        """
        flow = self.processor.compute_optical_flow(window)

        velocity = np.sqrt(flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2)
        v_mean = np.mean(velocity, axis=(1, 2))

        acceleration = np.diff(v_mean)

        energy = float(np.sum(v_mean**2) + np.sum(acceleration**2))
        return energy

    def _compute_scene_change(self, window: VideoWindow) -> float:
        """
        Scene change magnitude: ||z_end - z_start||

        Physics: Phase transition magnitude
        """
        hist_start = self._frame_histogram(window.frames[0])
        hist_end = self._frame_histogram(window.frames[-1])

        distance = float(np.linalg.norm(hist_end - hist_start))
        return distance

    def _frame_histogram(self, frame: np.ndarray, bins: int = 32) -> np.ndarray:
        """Compute normalized color histogram."""
        hist_r = np.histogram(frame[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(frame[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(frame[:, :, 2], bins=bins, range=(0, 256))[0]
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float64)
        return hist / (hist.sum() + 1e-6)

    def _compute_interaction(self, window: VideoWindow) -> float:
        """
        Interaction density: proximity changes between objects.

        Physics: Multi-agent coupling strength

        Uses edge detection as a proxy for object presence.
        """
        edges = []
        for frame in window.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 100, 200)
            edges.append(np.sum(edge > 0))

        interaction = float(np.std(edges))
        return interaction

    def _compute_uncertainty(self, window: VideoWindow) -> float:
        """
        Model uncertainty: H(p(y|x))

        Physics: Information entropy

        Uses variance in pixel intensity as proxy.
        """
        variances = [float(np.var(frame)) for frame in window.frames]
        uncertainty = float(np.mean(variances))
        return uncertainty

    def compute_energy(self, features: List[Dict[str, float]]) -> np.ndarray:
        """
        E(W_i) = sum alpha_k * phi_k(W_i)

        Physics: Hamiltonian energy functional
        """
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

    def adaptive_threshold(self, energies: np.ndarray, level: int) -> float:
        """
        tau_l = mu(E) + sigma_l * std(E)

        Adaptive based on energy distribution.
        """
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        sigma_mult = self.config.sigma_multipliers[level]

        return float(mean_energy + sigma_mult * std_energy)

    def sparse_select(
        self, candidates: List[VideoWindow]
    ) -> List[VideoWindow]:
        """
        Greedy sparse selection with diversity constraint.

        Solve: max sum S(W_i) - lambda * sum sim(W_i, W_j)
        """
        if len(candidates) <= self.config.top_k:
            return candidates

        features = self.extract_features(candidates, level=2)
        scores = self.compute_energy(features)

        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(self.config.top_k):
            if not remaining:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                score = scores[idx]

                if selected_indices:
                    max_sim = max(
                        self._temporal_similarity(
                            candidates[idx], candidates[sel_idx]
                        )
                        for sel_idx in selected_indices
                    )
                    score -= self.config.diversity_weight * max_sim

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _temporal_similarity(
        self, w1: VideoWindow, w2: VideoWindow
    ) -> float:
        """
        Temporal similarity: penalize adjacent windows.

        sim(w1, w2) = exp(-|t1 - t2| / sigma)
        """
        time_diff = abs(w1.start_time - w2.start_time)
        sigma = 10.0
        return float(np.exp(-time_diff / sigma))
