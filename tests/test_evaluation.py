"""Tests for evaluation metrics."""

import numpy as np
import pytest

from event_discovery.core.video_processor import VideoWindow
from event_discovery.evaluation import compute_metrics, temporal_iou


class TestTemporalIoU:
    def test_perfect_overlap(self):
        gt = {"start_time": 10.0, "end_time": 12.0}
        det = VideoWindow(
            start_time=10.0, end_time=12.0,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )
        assert temporal_iou(gt, det) == pytest.approx(1.0)

    def test_no_overlap(self):
        gt = {"start_time": 10.0, "end_time": 12.0}
        det = VideoWindow(
            start_time=20.0, end_time=22.0,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )
        assert temporal_iou(gt, det) == pytest.approx(0.0)

    def test_partial_overlap(self):
        gt = {"start_time": 10.0, "end_time": 12.0}
        det = VideoWindow(
            start_time=11.0, end_time=13.0,
            frames=np.zeros((1, 1, 1, 3), dtype=np.uint8),
            frame_indices=[0],
        )
        # intersection = 1.0, union = 3.0
        assert temporal_iou(gt, det) == pytest.approx(1.0 / 3.0)


class TestComputeMetrics:
    def _make_window(self, start, end):
        return VideoWindow(
            start_time=start, end_time=end,
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
            PureOptimizationMethod,
            OptimizationConfig,
        )

        config = OptimizationConfig(top_k=3)
        method = PureOptimizationMethod(config)
        features = method.extract_all_features(sample_windows[:5])
        assert len(features) == 5

        scores = method.compute_scores(features)
        assert len(scores) == 5

        selected = method.sparse_select(sample_windows[:5], scores)
        assert len(selected) <= 3
