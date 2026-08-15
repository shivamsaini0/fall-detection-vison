"""YOLO pose detection and BoT-SORT tracking wrapper."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from src.utils import COLOR_GREEN, COLOR_WHITE, draw_bbox, draw_skeleton

logger = logging.getLogger(__name__)


@dataclass
class DetectedPerson:
    """A single detected person in a frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    keypoints: np.ndarray  # shape (17, 3) — [x, y, confidence]
    confidence: float

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


@dataclass
class FrameResult:
    """Detection results for a single frame."""

    frame_num: int
    persons: list[DetectedPerson] = field(default_factory=list)

    @property
    def num_persons(self) -> int:
        return len(self.persons)

    @property
    def track_ids(self) -> list[int]:
        return [p.track_id for p in self.persons]


class PersonDetector:
    """YOLO pose detection with BoT-SORT tracking.

    This class wraps the Ultralytics YOLO pose model and provides
    a clean interface for detecting and tracking persons across frames.
    """

    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        device: str | None = None,
        confidence_threshold: float = 0.5,
        tracker_config: str | Path = "configs/botsort.yaml",
    ):
        """Initialize the person detector.

        Args:
            model_name: YOLO pose model name or path.
            device: Device to run on ('cuda', 'cpu', or None for auto).
            confidence_threshold: Minimum detection confidence.
            tracker_config: Path to BoT-SORT tracker configuration.
        """
        self.confidence_threshold = confidence_threshold
        self.tracker_config = str(tracker_config)

        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading YOLO model '{model_name}' on device '{self.device}'")
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_name}'. "
                f"Ensure you have internet access for first download. Error: {e}"
            ) from e

        logger.info("PersonDetector initialized successfully")

    def detect_frame(self, frame: np.ndarray, frame_num: int = 0) -> FrameResult:
        """Run detection + tracking on a single frame.

        Args:
            frame: BGR image as numpy array.
            frame_num: Frame number for the result.

        Returns:
            FrameResult with detected persons.
        """
        try:
            results = self.model.track(
                source=frame,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
                device=self.device,
                conf=self.confidence_threshold,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"GPU OOM on frame {frame_num}. Switching to CPU permanently "
                f"and resetting tracker to avoid device mismatch."
            )
            self.device = "cpu"
            self.reset_tracker()
            results = self.model.track(
                source=frame,
                tracker=self.tracker_config,
                persist=True,
                verbose=False,
                device="cpu",
                conf=self.confidence_threshold,
            )

        result = results[0] if results else None
        frame_result = FrameResult(frame_num=frame_num)

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return frame_result

        boxes = result.boxes
        keypoints_data = result.keypoints

        for i in range(len(boxes)):
            # Get track ID (BoT-SORT assigns these)
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())
            else:
                # No tracking ID available (first frame or lost track)
                track_id = -1

            # Get bounding box
            bbox = boxes.xyxy[i].cpu().numpy()
            bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

            # Get detection confidence
            conf = float(boxes.conf[i].item())

            # Get keypoints (17 keypoints, each with x, y, confidence)
            if keypoints_data is not None and i < len(keypoints_data):
                kps = keypoints_data[i].data.cpu().numpy()
                if kps.ndim == 3:
                    kps = kps[0]  # Remove batch dimension if present
                # Ensure shape is (17, 3)
                if kps.shape[0] != 17 or kps.shape[1] < 2:
                    logger.warning(f"Unexpected keypoint shape {kps.shape} for person {i}")
                    kps = np.zeros((17, 3), dtype=np.float32)
                elif kps.shape[1] == 2:
                    # No confidence scores, add ones
                    kps = np.hstack([kps, np.ones((17, 1), dtype=np.float32)])
            else:
                kps = np.zeros((17, 3), dtype=np.float32)

            person = DetectedPerson(
                track_id=track_id,
                bbox=bbox,
                keypoints=kps,
                confidence=conf,
            )
            frame_result.persons.append(person)

        return frame_result

    def reset_tracker(self):
        """Reset the tracker state. Call between different videos."""
        self.model.predictor = None

    def draw_results(
        self,
        frame: np.ndarray,
        frame_result: FrameResult,
        draw_skeletons: bool = True,
        draw_boxes: bool = True,
        draw_ids: bool = True,
    ) -> np.ndarray:
        """Draw detection results on a frame.

        Args:
            frame: BGR image to draw on (modified in place).
            frame_result: Detection results for this frame.
            draw_skeletons: Whether to draw skeleton connections.
            draw_boxes: Whether to draw bounding boxes.
            draw_ids: Whether to draw track ID labels.

        Returns:
            The annotated frame.
        """
        for person in frame_result.persons:
            color = COLOR_GREEN

            if draw_skeletons:
                draw_skeleton(frame, person.keypoints, color=color)

            label = None
            if draw_ids and person.track_id >= 0:
                label = f"ID:{person.track_id} {person.confidence:.2f}"

            if draw_boxes:
                draw_bbox(frame, person.bbox, color=COLOR_WHITE, label=label)

        return frame
