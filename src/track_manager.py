"""Per-track sliding window buffer management for temporal classification."""

import logging
from collections import deque

import numpy as np

from src.feature_extraction import FeatureVector

logger = logging.getLogger(__name__)


class TrackState:
    """State for a single tracked person."""

    def __init__(self, track_id: int, window_size: int = 30):
        self.track_id = track_id
        self.window_size = window_size
        self.features: deque[FeatureVector] = deque(maxlen=window_size)
        self.last_seen_frame: int = 0
        self.prev_cog_height: float | None = None
        self.classification: str = "normal"
        self.classification_confidence: float = 0.0
        self.total_frames_seen: int = 0

    def add_features(self, fv: FeatureVector, frame_num: int) -> None:
        """Add a feature vector to this track's buffer."""
        self.features.append(fv)
        self.last_seen_frame = frame_num
        self.prev_cog_height = fv.cog_height
        self.total_frames_seen += 1

    def get_sequence(self) -> np.ndarray | None:
        """Get the feature sequence as a numpy array.

        Returns:
            Array of shape (window_size, NUM_FEATURES) if buffer is full,
            None otherwise.
        """
        if len(self.features) < self.window_size:
            return None
        return np.array([fv.to_array() for fv in self.features], dtype=np.float32)

    def is_ready(self) -> bool:
        """Check if the buffer has enough frames for classification."""
        return len(self.features) >= self.window_size

    @property
    def buffer_fill(self) -> float:
        """Fraction of buffer that is filled (0.0 to 1.0)."""
        return len(self.features) / self.window_size


class TrackManager:
    """Manages per-track sliding window buffers for all tracked persons.

    Handles track creation, feature buffering, stale track cleanup,
    and provides sequences ready for LSTM classification.
    """

    def __init__(
        self,
        window_size: int = 30,
        stale_threshold: int = 90,
    ):
        """Initialize the track manager.

        Args:
            window_size: Number of frames to buffer per track for classification.
            stale_threshold: Remove tracks not seen for this many frames.
        """
        self.window_size = window_size
        self.stale_threshold = stale_threshold
        self.tracks: dict[int, TrackState] = {}
        self._total_tracks_created = 0

    def update(self, track_id: int, features: FeatureVector, frame_num: int) -> None:
        """Update a track with new features.

        Creates the track if it doesn't exist.

        Args:
            track_id: BoT-SORT track ID.
            features: Extracted features for this frame.
            frame_num: Current frame number.
        """
        if track_id < 0:
            return  # Skip untracked detections

        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id, self.window_size)
            self._total_tracks_created += 1
            logger.debug(f"New track created: {track_id}")

        self.tracks[track_id].add_features(features, frame_num)

    def get_sequence(self, track_id: int) -> np.ndarray | None:
        """Get the feature sequence for a track.

        Returns:
            Array of shape (window_size, NUM_FEATURES) if ready, None otherwise.
        """
        if track_id not in self.tracks:
            return None
        return self.tracks[track_id].get_sequence()

    def get_prev_cog_height(self, track_id: int) -> float | None:
        """Get the previous CoG height for velocity computation."""
        if track_id not in self.tracks:
            return None
        return self.tracks[track_id].prev_cog_height

    def set_classification(
        self, track_id: int, classification: str, confidence: float
    ) -> None:
        """Set the classification result for a track."""
        if track_id in self.tracks:
            self.tracks[track_id].classification = classification
            self.tracks[track_id].classification_confidence = confidence

    def get_classification(self, track_id: int) -> tuple[str, float]:
        """Get the classification result for a track.

        Returns:
            Tuple of (classification_label, confidence).
        """
        if track_id not in self.tracks:
            return "normal", 0.0
        track = self.tracks[track_id]
        return track.classification, track.classification_confidence

    def cleanup(self, current_frame: int) -> list[int]:
        """Remove stale tracks that haven't been seen recently.

        Args:
            current_frame: Current frame number.

        Returns:
            List of removed track IDs.
        """
        stale_ids = [
            tid
            for tid, track in self.tracks.items()
            if current_frame - track.last_seen_frame > self.stale_threshold
        ]
        for tid in stale_ids:
            del self.tracks[tid]
            logger.debug(f"Removed stale track: {tid}")
        return stale_ids

    def get_ready_tracks(self) -> list[int]:
        """Get track IDs that have full buffers ready for classification."""
        return [tid for tid, track in self.tracks.items() if track.is_ready()]

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self.tracks)

    @property
    def total_tracks_created(self) -> int:
        """Total number of tracks ever created."""
        return self._total_tracks_created

    def reset(self) -> None:
        """Clear all tracks. Call between videos."""
        self.tracks.clear()
        self._total_tracks_created = 0
