"""Video reading and writing utilities with proper resource management."""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

CODEC_MAP = {
    ".mp4": "mp4v",
    ".avi": "XVID",
    ".mov": "mp4v",
    ".mkv": "mp4v",
}


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str


def validate_video_path(path: str | Path, must_exist: bool = True) -> Path:
    """Validate a video file path.

    Args:
        path: Path to the video file.
        must_exist: If True, verify the file exists.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If must_exist and file doesn't exist.
        ValueError: If extension is unsupported.
    """
    path = Path(path)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


class VideoReader:
    """Context manager for reading video files frame by frame.

    Usage:
        with VideoReader("input.mp4") as reader:
            for frame_num, frame in reader:
                process(frame)
    """

    def __init__(self, path: str | Path):
        self.path = validate_video_path(path, must_exist=True)
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None
        self._frame_num = 0

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Failed to open video file: {self.path}")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        if fps <= 0:
            fps = 30.0  # Fallback for videos with missing FPS metadata

        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {self.path}")

        duration = total_frames / fps if fps > 0 else 0.0

        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec,
        )
        self._frame_num = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        return False

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened. Use as context manager.")
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        frame_num = self._frame_num
        self._frame_num += 1
        return frame_num, frame

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader not opened. Use as context manager.")
        return self._metadata


class VideoWriter:
    """Context manager for writing video files.

    Usage:
        with VideoWriter("output.mp4", width=1920, height=1080, fps=30.0) as writer:
            writer.write(frame)
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        self.path = validate_video_path(path, must_exist=False)
        self.width = width
        self.height = height
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0

    def __enter__(self) -> "VideoWriter":
        os.makedirs(self.path.parent, exist_ok=True)

        codec_str = CODEC_MAP.get(self.path.suffix.lower(), "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*codec_str)

        self._writer = cv2.VideoWriter(
            str(self.path), fourcc, self.fps, (self.width, self.height)
        )
        if not self._writer.isOpened():
            raise IOError(f"Failed to create video writer for: {self.path}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        # Convert to H.264 for browser/app compatibility
        if exc_type is None and self._frame_count > 0:
            self._convert_to_h264()

        return False

    def _convert_to_h264(self):
        """Convert output video to H.264 codec using ffmpeg for browser compatibility."""
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found, skipping H.264 conversion")
            return

        tmp_path = self.path.with_suffix(".tmp.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(self.path),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-an",
                    str(tmp_path),
                ],
                capture_output=True,
                timeout=300,
            )
            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                tmp_path.replace(self.path)
                logger.debug(f"Converted to H.264: {self.path}")
            else:
                logger.warning("H.264 conversion produced empty file, keeping original")
                if tmp_path.exists():
                    tmp_path.unlink()
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(f"H.264 conversion failed: {e}, keeping original")
            if tmp_path.exists():
                tmp_path.unlink()

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame to the output video."""
        if self._writer is None:
            raise RuntimeError("VideoWriter not opened. Use as context manager.")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            logger.warning(
                f"Frame size ({frame.shape[1]}x{frame.shape[0]}) doesn't match "
                f"writer size ({self.width}x{self.height}). Resizing."
            )
            frame = cv2.resize(frame, (self.width, self.height))

        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count


def format_timestamp(frame_num: int, fps: float) -> str:
    """Convert frame number to HH:MM:SS.ms timestamp."""
    if fps <= 0 or frame_num < 0:
        return "00:00:00.000"
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS.s string."""
    if seconds < 0:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ImageSequenceReader:
    """Context manager for reading a folder of numbered images as a video.

    Supports datasets like UR Fall that store frames as individual PNG/JPG files.

    Usage:
        with ImageSequenceReader("path/to/frames/", fps=30.0) as reader:
            for frame_num, frame in reader:
                process(frame)
    """

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, directory: str | Path, fps: float = 30.0):
        self.directory = Path(directory)
        self.fps = fps
        self._image_paths: list[Path] = []
        self._metadata: VideoMetadata | None = None
        self._index = 0

    def __enter__(self) -> "ImageSequenceReader":
        if not self.directory.exists() or not self.directory.is_dir():
            raise FileNotFoundError(f"Image sequence directory not found: {self.directory}")

        # Collect and sort image files
        self._image_paths = sorted(
            f for f in self.directory.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        if not self._image_paths:
            raise ValueError(f"No image files found in: {self.directory}")

        # Read first image to get dimensions
        first_frame = cv2.imread(str(self._image_paths[0]))
        if first_frame is None:
            raise IOError(f"Failed to read first image: {self._image_paths[0]}")

        height, width = first_frame.shape[:2]
        total_frames = len(self._image_paths)
        duration = total_frames / self.fps if self.fps > 0 else 0.0

        self._metadata = VideoMetadata(
            width=width,
            height=height,
            fps=self.fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec="PNG",
        )
        self._index = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._image_paths = []
        return False

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self._index >= len(self._image_paths):
            raise StopIteration

        img_path = self._image_paths[self._index]
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Failed to read image {img_path}, skipping")
            self._index += 1
            return self.__next__()

        frame_num = self._index
        self._index += 1
        return frame_num, frame

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("ImageSequenceReader not opened. Use as context manager.")
        return self._metadata
