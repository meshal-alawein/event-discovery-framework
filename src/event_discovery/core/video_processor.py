"""
Core video processing utilities for event discovery.
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VideoWindow:
    """Represents a temporal window in a video."""
    start_time: float
    end_time: float
    frames: np.ndarray  # Shape: (T, H, W, C)
    frame_indices: List[int]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)


class VideoProcessor:
    """
    Base class for processing long videos into temporal windows.
    """

    def __init__(self, window_size: float = 2.0, stride: float = 1.0):
        """
        Args:
            window_size: Duration of each window in seconds
            stride: Stride between windows in seconds
        """
        self.window_size = window_size
        self.stride = stride

    def chunk_video(self, video_path: str) -> List[VideoWindow]:
        """
        Discretize video into temporal windows.

        Args:
            video_path: Path to video file

        Returns:
            List of VideoWindow objects
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_window = int(self.window_size * fps)
        frames_per_stride = int(self.stride * fps)

        windows = []
        start_frame = 0

        while start_frame + frames_per_window <= total_frames:
            end_frame = start_frame + frames_per_window

            frame_list = []
            frame_indices = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in range(start_frame, end_frame):
                ret, frame = cap.read()
                if ret:
                    frame_list.append(frame)
                    frame_indices.append(i)

            if frame_list:
                frames_array = np.stack(frame_list)
                window = VideoWindow(
                    start_time=start_frame / fps,
                    end_time=end_frame / fps,
                    frames=frames_array,
                    frame_indices=frame_indices,
                )
                windows.append(window)

            start_frame += frames_per_stride

        cap.release()
        logger.info("Chunked video into %d windows", len(windows))
        return windows

    def load_annotations(self, annotation_path: str) -> List[dict]:
        """
        Load ground truth event annotations.

        Args:
            annotation_path: Path to JSON annotation file

        Returns:
            List of event dictionaries with start_time, end_time, label
        """
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        return annotations["events"]

    def compute_optical_flow(self, window: VideoWindow) -> np.ndarray:
        """
        Compute dense optical flow for a window.

        Args:
            window: VideoWindow object

        Returns:
            Optical flow array of shape (T-1, H, W, 2)
        """
        flows = []
        gray_prev = cv2.cvtColor(window.frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, len(window.frames)):
            gray_curr = cv2.cvtColor(window.frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev,
                gray_curr,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            flows.append(flow)
            gray_prev = gray_curr

        return np.stack(flows)

    def downsample_window(
        self, window: VideoWindow, target_frames: int = 8
    ) -> np.ndarray:
        """
        Downsample window to fixed number of frames for embedding.

        Args:
            window: VideoWindow object
            target_frames: Number of frames to sample

        Returns:
            Downsampled frames array
        """
        num_frames = len(window.frames)
        if num_frames <= target_frames:
            return window.frames

        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        return window.frames[indices]


def visualize_detections(
    video_path: str,
    detected_windows: List[VideoWindow],
    output_path: str,
    annotations: Optional[List[dict]] = None,
):
    """
    Create visualization video with detected events highlighted.

    Args:
        video_path: Original video path
        detected_windows: List of detected event windows
        output_path: Path for output video
        annotations: Optional ground truth annotations
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_times = set()
    for window in detected_windows:
        for t in np.arange(window.start_time, window.end_time, 1 / fps):
            detected_times.add(round(t, 3))

    gt_times = set()
    if annotations:
        for event in annotations:
            for t in np.arange(event["start_time"], event["end_time"], 1 / fps):
                gt_times.add(round(t, 3))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = round(frame_idx / fps, 3)

        if current_time in detected_times:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 0), 10)
            cv2.putText(
                frame,
                "DETECTED EVENT",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )

        if current_time in gt_times:
            cv2.putText(
                frame,
                "GT EVENT",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    logger.info("Visualization saved to %s", output_path)
