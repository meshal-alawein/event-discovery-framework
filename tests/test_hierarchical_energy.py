"""Tests for hierarchical energy method."""

import numpy as np
import pytest

from event_discovery.methods.hierarchical_energy import (
    HierarchicalEnergyMethod,
    EnergyConfig,
)


class TestEnergyConfig:
    def test_default_weights_sum_to_one(self):
        config = EnergyConfig()
        config.normalize_weights()
        total = (
            config.weight_motion
            + config.weight_interaction
            + config.weight_scene_change
            + config.weight_uncertainty
        )
        assert total == pytest.approx(1.0)

    def test_custom_weights_normalize(self):
        config = EnergyConfig(
            weight_motion=1.0,
            weight_interaction=1.0,
            weight_scene_change=1.0,
            weight_uncertainty=1.0,
        )
        config.normalize_weights()
        assert config.weight_motion == pytest.approx(0.25)

    def test_num_levels(self):
        config = EnergyConfig()
        assert config.num_levels == 3


class TestHierarchicalEnergyMethod:
    def test_init_default(self):
        method = HierarchicalEnergyMethod()
        assert method.config.top_k == 10

    def test_motion_energy(self, sample_window):
        method = HierarchicalEnergyMethod()
        energy = method._compute_motion_energy(sample_window)
        assert isinstance(energy, float)
        assert energy >= 0

    def test_scene_change(self, sample_window):
        method = HierarchicalEnergyMethod()
        change = method._compute_scene_change(sample_window)
        assert isinstance(change, float)
        assert change >= 0

    def test_interaction(self, sample_window):
        method = HierarchicalEnergyMethod()
        interaction = method._compute_interaction(sample_window)
        assert isinstance(interaction, float)
        assert interaction >= 0

    def test_uncertainty(self, sample_window):
        method = HierarchicalEnergyMethod()
        uncertainty = method._compute_uncertainty(sample_window)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0

    def test_extract_features(self, sample_windows):
        method = HierarchicalEnergyMethod()
        features = method.extract_features(sample_windows[:5], level=0)
        assert len(features) == 5
        assert "motion" in features[0]
        assert "scene_change" in features[0]

    def test_compute_energy(self, sample_windows):
        method = HierarchicalEnergyMethod()
        features = method.extract_features(sample_windows[:5], level=0)
        energies = method.compute_energy(features)
        assert len(energies) == 5
        assert energies.dtype == np.float64

    def test_adaptive_threshold(self):
        method = HierarchicalEnergyMethod()
        energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold = method.adaptive_threshold(energies, level=0)
        # mean=3.0, std~1.414, sigma_mult=2.0 -> 3.0 + 2.0*1.414 ≈ 5.83
        assert threshold > np.mean(energies)

    def test_hierarchical_filter_reduces_candidates(self, sample_windows):
        config = EnergyConfig(sigma_multipliers=[1.0, 0.5, 0.0])
        method = HierarchicalEnergyMethod(config)
        candidates = method.hierarchical_filter(sample_windows)
        assert len(candidates) <= len(sample_windows)

    def test_sparse_select_respects_top_k(self, sample_windows):
        config = EnergyConfig(top_k=3)
        method = HierarchicalEnergyMethod(config)
        selected = method.sparse_select(sample_windows)
        assert len(selected) <= 3

    def test_temporal_similarity(self, sample_windows):
        method = HierarchicalEnergyMethod()
        sim = method._temporal_similarity(sample_windows[0], sample_windows[1])
        assert 0.0 <= sim <= 1.0
        # Adjacent windows should have higher similarity
        sim_far = method._temporal_similarity(sample_windows[0], sample_windows[-1])
        assert sim > sim_far
