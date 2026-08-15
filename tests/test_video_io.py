"""Tests for video I/O utilities."""

import cv2
import numpy as np
import pytest

from src.video_io import (
    VideoReader,
    VideoWriter,
    format_duration,
    format_timestamp,
    validate_video_path,
)


@pytest.fixture
def sample_video(tmp_path):
    """Create a small sample video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    width, height, fps = 640, 480, 30.0
    num_frames = 90  # 3 seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a frame with varying color to make it non-trivial
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 2 % 256  # varying blue
        frame[:, :, 1] = 128  # constant green
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def empty_video(tmp_path):
    """Create a video file path that doesn't exist."""
    return tmp_path / "nonexistent.mp4"


class TestValidateVideoPath:
    def test_valid_mp4(self, sample_video):
        result = validate_video_path(sample_video)
        assert result == sample_video

    def test_valid_extensions(self, tmp_path):
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            path = tmp_path / f"test{ext}"
            path.touch()
            result = validate_video_path(path)
            assert result.suffix == ext

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "test.gif"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported video format"):
            validate_video_path(path)

    def test_nonexistent_file(self, tmp_path):
        path = tmp_path / "missing.mp4"
        with pytest.raises(FileNotFoundError):
            validate_video_path(path, must_exist=True)

    def test_nonexistent_no_check(self, tmp_path):
        path = tmp_path / "missing.mp4"
        result = validate_video_path(path, must_exist=False)
        assert result == path


class TestVideoReader:
    def test_read_all_frames(self, sample_video):
        with VideoReader(sample_video) as reader:
            frames = list(reader)
        assert len(frames) == 90

    def test_metadata(self, sample_video):
        with VideoReader(sample_video) as reader:
            meta = reader.metadata
            assert meta.width == 640
            assert meta.height == 480
            assert meta.fps == pytest.approx(30.0, abs=1.0)
            assert meta.total_frames == 90
            assert meta.duration_seconds == pytest.approx(3.0, abs=0.5)

    def test_frame_numbering(self, sample_video):
        with VideoReader(sample_video) as reader:
            first_num, first_frame = next(iter(reader))
            assert first_num == 0
            assert first_frame.shape == (480, 640, 3)

    def test_nonexistent_file_raises(self, empty_video):
        with pytest.raises(FileNotFoundError):
            with VideoReader(empty_video) as _reader:
                pass

    def test_context_manager_cleanup(self, sample_video):
        reader = VideoReader(sample_video)
        with reader:
            _ = next(iter(reader))
        # After context manager, cap should be released
        assert reader._cap is None

    def test_metadata_before_open_raises(self, sample_video):
        reader = VideoReader(sample_video)
        with pytest.raises(RuntimeError):
            _ = reader.metadata


class TestVideoWriter:
    def test_write_and_read_back(self, tmp_path):
        output_path = tmp_path / "output.mp4"
        width, height, fps = 320, 240, 25.0
        num_frames = 30

        # Write frames
        with VideoWriter(output_path, width, height, fps) as writer:
            for _ in range(num_frames):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                writer.write(frame)
            assert writer.frame_count == num_frames

        # Read back
        assert output_path.exists()
        with VideoReader(output_path) as reader:
            frames = list(reader)
            assert len(frames) == num_frames
            assert reader.metadata.width == width
            assert reader.metadata.height == height

    def test_auto_resize(self, tmp_path):
        output_path = tmp_path / "output.mp4"
        with VideoWriter(output_path, 320, 240, 30.0) as writer:
            # Write a frame with different dimensions — should auto-resize
            big_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(big_frame)
            assert writer.frame_count == 1

    def test_creates_parent_directory(self, tmp_path):
        output_path = tmp_path / "subdir" / "output.mp4"
        with VideoWriter(output_path, 320, 240, 30.0) as writer:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        assert output_path.exists()


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0, 30.0) == "00:00:00.000"

    def test_one_second(self):
        assert format_timestamp(30, 30.0) == "00:00:01.000"

    def test_one_minute(self):
        assert format_timestamp(1800, 30.0) == "00:01:00.000"

    def test_one_hour(self):
        assert format_timestamp(108000, 30.0) == "01:00:00.000"

    def test_fractional(self):
        assert format_timestamp(15, 30.0) == "00:00:00.500"

    def test_zero_fps_fallback(self):
        assert format_timestamp(100, 0.0) == "00:00:00.000"


class TestFormatDuration:
    def test_zero(self):
        assert format_duration(0.0) == "00:00:00"

    def test_one_minute(self):
        assert format_duration(60.0) == "00:01:00"

    def test_complex(self):
        assert format_duration(3661.0) == "01:01:01"
