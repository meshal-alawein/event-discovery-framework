"""Core infrastructure for event discovery."""

from .video_processor import VideoWindow, VideoProcessor, visualize_detections
from .base import BaseEventDetector
from .features import (
    compute_color_histogram,
    compute_edge_density_variance,
    compute_pixel_variance,
    compute_pixel_entropy,
    normalize_features_batch,
    temporal_similarity,
    greedy_diverse_select,
)

__all__ = [
    "VideoWindow",
    "VideoProcessor",
    "visualize_detections",
    "BaseEventDetector",
    "compute_color_histogram",
    "compute_edge_density_variance",
    "compute_pixel_variance",
    "compute_pixel_entropy",
    "normalize_features_batch",
    "temporal_similarity",
    "greedy_diverse_select",
]
