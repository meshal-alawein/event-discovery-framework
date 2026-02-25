"""Tests for core video processor."""

import pytest

from event_discovery.core.video_processor import VideoProcessor


class TestVideoWindow:
    def test_duration(self, sample_window):
        assert sample_window.duration == pytest.approx(2.0)

    def test_num_frames(self, sample_window):
        assert sample_window.num_frames == 10

    def test_frames_shape(self, sample_window):
        assert sample_window.frames.shape == (10, 64, 64, 3)


class TestVideoProcessor:
    def test_init_defaults(self):
        proc = VideoProcessor()
        assert proc.window_size == 2.0
        assert proc.stride == 1.0

    def test_init_custom(self):
        proc = VideoProcessor(window_size=5.0, stride=2.5)
        assert proc.window_size == 5.0
        assert proc.stride == 2.5

    def test_compute_optical_flow(self, sample_window):
        proc = VideoProcessor()
        flow = proc.compute_optical_flow(sample_window)
        # T-1 flow frames, same H/W, 2 channels (dx, dy)
        assert flow.shape == (9, 64, 64, 2)

    def test_downsample_window(self, sample_window):
        proc = VideoProcessor()
        downsampled = proc.downsample_window(sample_window, target_frames=4)
        assert downsampled.shape[0] == 4
        assert downsampled.shape[1:] == (64, 64, 3)

    def test_downsample_no_op_when_fewer_frames(self, sample_window):
        proc = VideoProcessor()
        downsampled = proc.downsample_window(sample_window, target_frames=20)
        assert len(downsampled) == 10

    def test_chunk_video_invalid_path(self):
        proc = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            proc.chunk_video("/nonexistent/video.mp4")
