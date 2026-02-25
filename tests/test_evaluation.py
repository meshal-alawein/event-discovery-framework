"""Tests for evaluation metrics and shared features."""

import json

import numpy as np
import pytest

from event_discovery.core.features import (
    compute_color_histogram,
    compute_edge_density_variance,
    compute_pixel_entropy,
    compute_pixel_variance,
    greedy_diverse_select,
    normalize_features_batch,
    temporal_similarity,
)
from event_discovery.core.video_processor import VideoWindow
from event_discovery.evaluation import (
    AnnotationError,
    compute_metrics,
    load_ground_truth,
    temporal_iou,
)


class TestTemporalIoU:
    def test_perfect_overlap(self):
        gt = {"start_time": 10.0, "end_time": 12.0}
        det = VideoWindow(
            start_time=10.0,
            end_time=12.0,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )
        assert temporal_iou(gt, det) == pytest.approx(1.0)

    def test_no_overlap(self):
        gt = {"start_time": 10.0, "end_time": 12.0}
        det = VideoWindow(
            start_time=20.0,
            end_time=22.0,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )
        assert temporal_iou(gt, det) == pytest.approx(0.0)

    def test_partial_overlap(self):
        gt = {"start_time": 10.0, "end_time": 12.0}
        det = VideoWindow(
            start_time=11.0,
            end_time=13.0,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )
        assert temporal_iou(gt, det) == pytest.approx(1.0 / 3.0)


class TestComputeMetrics:
    def _make_window(self, start, end):
        return VideoWindow(
            start_time=start,
            end_time=end,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )

    def test_perfect_detection(self):
        gt = [{"start_time": 10.0, "end_time": 12.0}]
        det = [self._make_window(10.0, 12.0)]
        metrics = compute_metrics(det, gt, iou_threshold=0.5)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_no_detections(self):
        gt = [{"start_time": 10.0, "end_time": 12.0}]
        metrics = compute_metrics([], gt)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["f1"] == pytest.approx(0.0)

    def test_false_positive(self):
        gt = [{"start_time": 10.0, "end_time": 12.0}]
        det = [
            self._make_window(10.0, 12.0),
            self._make_window(50.0, 52.0),
        ]
        metrics = compute_metrics(det, gt, iou_threshold=0.5)
        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["fn"] == 0
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_empty_ground_truth(self):
        det = [self._make_window(10.0, 12.0)]
        metrics = compute_metrics(det, [])
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)


class TestLoadGroundTruth:
    def test_valid_file(self, tmp_path):
        data = {"events": [{"start_time": 1.0, "end_time": 2.0, "label": "test"}]}
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(data))
        events = load_ground_truth(str(path))
        assert len(events) == 1

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_ground_truth("/nonexistent/file.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        with pytest.raises(AnnotationError, match="Invalid JSON"):
            load_ground_truth(str(path))

    def test_missing_events_key(self, tmp_path):
        path = tmp_path / "no_events.json"
        path.write_text(json.dumps({"data": []}))
        with pytest.raises(AnnotationError, match="missing 'events' key"):
            load_ground_truth(str(path))


class TestSharedFeatures:
    def test_color_histogram_shape(self, sample_window):
        hist = compute_color_histogram(sample_window.frames[0], bins=32)
        assert hist.shape == (96,)
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_edge_density_variance(self, sample_window):
        val = compute_edge_density_variance(sample_window.frames)
        assert isinstance(val, float)
        assert val >= 0

    def test_pixel_variance(self, sample_window):
        val = compute_pixel_variance(sample_window.frames)
        assert isinstance(val, float)
        assert val >= 0

    def test_pixel_entropy(self, sample_window):
        val = compute_pixel_entropy(sample_window.frames)
        assert isinstance(val, float)
        assert val >= 0

    def test_normalize_features_batch(self):
        features = [
            {"a": 10.0, "b": 100.0},
            {"a": 20.0, "b": 200.0},
            {"a": 30.0, "b": 300.0},
        ]
        normalized = normalize_features_batch(features)
        assert len(normalized) == 3
        # After z-score, mean should be ~0
        a_vals = [f["a"] for f in normalized]
        assert np.mean(a_vals) == pytest.approx(0.0, abs=1e-5)

    def test_normalize_empty(self):
        assert normalize_features_batch([]) == []

    def test_temporal_similarity_same_time(self):
        assert temporal_similarity(5.0, 5.0) == pytest.approx(1.0)

    def test_temporal_similarity_far(self):
        sim = temporal_similarity(0.0, 100.0, sigma=10.0)
        assert sim < 0.001

    def test_greedy_diverse_select_top_k(self, sample_windows):
        scores = np.arange(len(sample_windows), dtype=float)
        selected = greedy_diverse_select(sample_windows, scores, top_k=3)
        assert len(selected) == 3

    def test_greedy_diverse_select_fewer_than_k(self):
        from event_discovery.core.video_processor import VideoWindow

        windows = [VideoWindow(0.0, 1.0, np.zeros((1, 1, 1, 3), dtype=np.uint8), [0])]
        scores = np.array([1.0])
        selected = greedy_diverse_select(windows, scores, top_k=5)
        assert len(selected) == 1


class TestGeometricOutlier:
    def test_process(self, sample_windows):
        from event_discovery.methods.geometric_outlier import GeometricOutlierMethod

        method = GeometricOutlierMethod(embedding_dim=4, top_k=3)
        embeddings = method.embed_windows(sample_windows[:10])
        assert embeddings.shape == (10, 4)

        scores = method.compute_outlier_scores(embeddings)
        assert len(scores) == 10


class TestOptimizationSparse:
    def test_process(self, sample_windows):
        from event_discovery.methods.optimization_sparse import (
            OptimizationConfig,
            PureOptimizationMethod,
        )

        config = OptimizationConfig(top_k=3)
        method = PureOptimizationMethod(config)
        features = method._extract_all_features(sample_windows[:5])
        assert len(features) == 5

        scores = method._compute_scores(features)
        assert len(scores) == 5
