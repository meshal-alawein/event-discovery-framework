"""
Base class for event detection methods.

Implements the Template Method pattern: subclasses only need
to define their scoring logic while sharing the common pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .video_processor import VideoWindow, VideoProcessor
from .features import greedy_diverse_select

logger = logging.getLogger(__name__)


class BaseEventDetector(ABC):
    """
    Abstract base class for all event detection methods.

    Subclasses must implement:
        _score_windows(windows) -> np.ndarray

    The common pipeline is: chunk -> score -> select.
    """

    def __init__(self, top_k: int = 10, diversity_weight: float = 0.5, sigma: float = 10.0):
        self.top_k = top_k
        self.diversity_weight = diversity_weight
        self.sigma = sigma
        self.processor = VideoProcessor()

    def process_video(self, video_path: str) -> List[VideoWindow]:
        """
        Main pipeline: chunk video -> score windows -> select top-k.

        Args:
            video_path: Path to video file

        Returns:
            List of detected event windows
        """
        windows = self.processor.chunk_video(video_path)
        logger.info("Chunked video into %d windows", len(windows))

        scores = self._score_windows(windows)
        logger.info("Computed scores for %d windows", len(windows))

        selected = self._select(windows, scores)
        logger.info("Selected %d events", len(selected))

        return selected

    @abstractmethod
    def _score_windows(self, windows: List[VideoWindow]) -> np.ndarray:
        """
        Compute a relevance score for each window.

        Args:
            windows: List of video windows

        Returns:
            Array of scores, shape (len(windows),)
        """

    def _select(self, windows: List[VideoWindow], scores: np.ndarray) -> List[VideoWindow]:
        """
        Select top-k diverse windows. Can be overridden for custom selection.
        """
        return greedy_diverse_select(
            candidates=windows,
            scores=scores,
            top_k=self.top_k,
            diversity_weight=self.diversity_weight,
            sigma=self.sigma,
        )
