"""Feature extraction from COCO skeleton keypoints for fall detection."""

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# COCO keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Number of features extracted per frame per person
NUM_FEATURES = 5


@dataclass
class FeatureVector:
    """Features extracted from a single person in a single frame.

    Attributes:
        body_angle: Angle between hip-midpoint→nose line and vertical (degrees, 0-180).
        cog_height: Normalized center of gravity height (0.0=top, 1.0=bottom).
        vertical_velocity: Change in CoG height from previous frame (-1.0 to 1.0).
        bbox_aspect_ratio: Bounding box width / height (clamped to 0-5).
        keypoint_confidence: Mean confidence of all 17 keypoints (0.0-1.0).
    """

    body_angle: float
    cog_height: float
    vertical_velocity: float
    bbox_aspect_ratio: float
    keypoint_confidence: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array of shape (5,)."""
        return np.array(
            [
                self.body_angle,
                self.cog_height,
                self.vertical_velocity,
                self.bbox_aspect_ratio,
                self.keypoint_confidence,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_array(arr: np.ndarray) -> "FeatureVector":
        """Create FeatureVector from numpy array of shape (5,)."""
        return FeatureVector(
            body_angle=float(arr[0]),
            cog_height=float(arr[1]),
            vertical_velocity=float(arr[2]),
            bbox_aspect_ratio=float(arr[3]),
            keypoint_confidence=float(arr[4]),
        )


class FeatureExtractor:
    """Extracts fall-relevant features from skeleton keypoints.

    Features:
        1. Body angle (0-180 degrees, normalized to 0-1)
        2. Center of gravity height (normalized by frame height)
        3. Vertical velocity (delta CoG between frames)
        4. Bounding box aspect ratio (width/height)
        5. Mean keypoint confidence
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        normalize_angle: bool = True,
    ):
        """Initialize the feature extractor.

        Args:
            confidence_threshold: Minimum keypoint confidence to consider valid.
            normalize_angle: If True, normalize body angle to 0-1 range.
        """
        self.confidence_threshold = confidence_threshold
        self.normalize_angle = normalize_angle

    def extract(
        self,
        keypoints: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_height: int,
        prev_cog_height: float | None = None,
    ) -> FeatureVector:
        """Extract features from a single person's keypoints and bbox.

        Args:
            keypoints: Array of shape (17, 3) with [x, y, confidence].
            bbox: Bounding box as (x1, y1, x2, y2).
            frame_height: Height of the video frame in pixels.
            prev_cog_height: CoG height from previous frame for velocity computation.

        Returns:
            FeatureVector with all 5 features.
        """
        body_angle = self._compute_body_angle(keypoints)
        cog_height = self._compute_cog_height(keypoints, frame_height)
        vertical_velocity = self._compute_velocity(cog_height, prev_cog_height)
        bbox_aspect_ratio = self._compute_bbox_aspect_ratio(bbox)
        keypoint_confidence = self._compute_mean_confidence(keypoints)

        return FeatureVector(
            body_angle=body_angle,
            cog_height=cog_height,
            vertical_velocity=vertical_velocity,
            bbox_aspect_ratio=bbox_aspect_ratio,
            keypoint_confidence=keypoint_confidence,
        )

    def _compute_body_angle(self, keypoints: np.ndarray) -> float:
        """Compute angle between hip-midpoint→nose line and vertical axis.

        Returns:
            Angle in degrees (0-180), or normalized to 0-1 if normalize_angle is True.
            Standing ≈ 0, lying ≈ 90 (or 0.5 normalized).
            Returns 0.0 if required keypoints are not visible.
        """
        nose = keypoints[NOSE]
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]

        # Check if we have the needed keypoints with sufficient confidence
        has_nose = nose[2] >= self.confidence_threshold
        has_left_hip = left_hip[2] >= self.confidence_threshold
        has_right_hip = right_hip[2] >= self.confidence_threshold

        if not has_nose or not (has_left_hip or has_right_hip):
            # Try shoulder midpoint as fallback for torso angle
            left_shoulder = keypoints[LEFT_SHOULDER]
            right_shoulder = keypoints[RIGHT_SHOULDER]
            has_ls = left_shoulder[2] >= self.confidence_threshold
            has_rs = right_shoulder[2] >= self.confidence_threshold

            if has_nose and (has_ls or has_rs):
                if has_ls and has_rs:
                    mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
                    mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                elif has_ls:
                    mid_x, mid_y = left_shoulder[0], left_shoulder[1]
                else:
                    mid_x, mid_y = right_shoulder[0], right_shoulder[1]
            else:
                return 0.0
        else:
            # Compute hip midpoint
            if has_left_hip and has_right_hip:
                mid_x = (left_hip[0] + right_hip[0]) / 2
                mid_y = (left_hip[1] + right_hip[1]) / 2
            elif has_left_hip:
                mid_x, mid_y = left_hip[0], left_hip[1]
            else:
                mid_x, mid_y = right_hip[0], right_hip[1]

        # Vector from midpoint to nose
        dx = float(nose[0] - mid_x)
        dy = float(mid_y - nose[1])  # Invert Y since image Y goes downward

        # Guard against NaN/Inf from corrupted keypoint coordinates
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return 0.0

        # Angle with vertical (upward direction)
        # abs(dx) so left/right lean gives same angle. Range: 0 (standing) to 180 (inverted)
        angle_rad = math.atan2(abs(dx), dy) if (dx != 0 or dy != 0) else 0.0
        angle_deg = math.degrees(angle_rad)

        # Clamp to 0-180
        angle_deg = max(0.0, min(180.0, angle_deg))

        if self.normalize_angle:
            return angle_deg / 180.0

        return angle_deg

    def _compute_cog_height(self, keypoints: np.ndarray, frame_height: int) -> float:
        """Compute normalized center of gravity height.

        Uses hip keypoints as CoG proxy, falls back to all visible keypoints.

        Returns:
            Normalized height (0.0=top, 1.0=bottom of frame).
            Returns 0.5 if no valid keypoints.
        """
        if frame_height <= 0:
            return 0.5

        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        has_left_hip = left_hip[2] >= self.confidence_threshold
        has_right_hip = right_hip[2] >= self.confidence_threshold

        if has_left_hip and has_right_hip:
            cog_y = (left_hip[1] + right_hip[1]) / 2
        elif has_left_hip:
            cog_y = left_hip[1]
        elif has_right_hip:
            cog_y = right_hip[1]
        else:
            # Fallback: use mean Y of all visible keypoints
            valid_mask = keypoints[:, 2] >= self.confidence_threshold
            if not np.any(valid_mask):
                return 0.5
            cog_y = np.mean(keypoints[valid_mask, 1])

        result = float(cog_y / frame_height)
        if not math.isfinite(result):
            return 0.5
        return float(np.clip(result, 0.0, 1.0))

    def _compute_velocity(
        self, cog_height: float, prev_cog_height: float | None
    ) -> float:
        """Compute vertical velocity as delta CoG height.

        Returns:
            Velocity clamped to [-1.0, 1.0]. Positive = moving down.
            Returns 0.0 for the first frame of a track.
        """
        if prev_cog_height is None:
            return 0.0
        velocity = cog_height - prev_cog_height
        return float(np.clip(velocity, -1.0, 1.0))

    def _compute_bbox_aspect_ratio(
        self, bbox: tuple[float, float, float, float]
    ) -> float:
        """Compute bounding box width/height ratio.

        Returns:
            Aspect ratio clamped to [0.0, 5.0].
            Standing person ≈ 0.3-0.5 (tall/narrow).
            Fallen person ≈ 1.5-3.0 (wide/short).
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return 0.0

        ratio = width / height
        return float(np.clip(ratio, 0.0, 5.0))

    def _compute_mean_confidence(self, keypoints: np.ndarray) -> float:
        """Compute mean confidence of all 17 keypoints.

        Returns:
            Mean confidence in [0.0, 1.0].
        """
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        return float(np.clip(np.mean(keypoints[:, 2]), 0.0, 1.0))
