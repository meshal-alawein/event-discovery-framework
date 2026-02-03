"""Shared test fixtures."""

import numpy as np
import pytest

from event_discovery.core.video_processor import VideoWindow


@pytest.fixture
def sample_window():
    """Create a simple test VideoWindow with synthetic frames."""
    np.random.seed(42)
    num_frames = 10
    h, w = 64, 64
    frames = np.random.randint(0, 256, (num_frames, h, w, 3), dtype=np.uint8)
    return VideoWindow(
        start_time=0.0,
        end_time=2.0,
        frames=frames,
        frame_indices=list(range(num_frames)),
    )


@pytest.fixture
def sample_windows():
    """Create multiple test VideoWindows with varying content."""
    np.random.seed(42)
    windows = []
    for i in range(20):
        num_frames = 10
        h, w = 64, 64
        # Make some windows have higher variance (potential events)
        if i in (5, 12, 17):
            # "Event" windows with high motion / scene change
            frames = np.random.randint(0, 256, (num_frames, h, w, 3), dtype=np.uint8)
            # Add a sudden change in the middle
            frames[5:] = np.clip(frames[5:] + 100, 0, 255).astype(np.uint8)
        else:
            # "Normal" windows with low variance
            base = np.random.randint(100, 156, (1, h, w, 3), dtype=np.uint8)
            noise = np.random.randint(-5, 6, (num_frames, h, w, 3))
            frames = np.clip(base + noise, 0, 255).astype(np.uint8)

        window = VideoWindow(
            start_time=i * 2.0,
            end_time=(i + 1) * 2.0,
            frames=frames,
            frame_indices=list(range(i * 10, i * 10 + num_frames)),
        )
        windows.append(window)
    return windows


@pytest.fixture
def sample_ground_truth():
    """Ground truth events matching the sample windows."""
    return [
        {"start_time": 10.0, "end_time": 12.0, "label": "event_1"},
        {"start_time": 24.0, "end_time": 26.0, "label": "event_2"},
        {"start_time": 34.0, "end_time": 36.0, "label": "event_3"},
    ]
