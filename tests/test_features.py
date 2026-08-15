"""Tests for feature extraction and track management."""

import math

import numpy as np
import pytest

from src.feature_extraction import NUM_FEATURES, FeatureExtractor, FeatureVector
from src.track_manager import TrackManager, TrackState

# --- Helper to create keypoints ---

def make_keypoints(
    nose=(320, 100, 0.9),
    left_shoulder=(280, 200, 0.9),
    right_shoulder=(360, 200, 0.9),
    left_hip=(280, 350, 0.9),
    right_hip=(360, 350, 0.9),
    default_conf=0.9,
) -> np.ndarray:
    """Create COCO keypoints array with specified positions."""
    kps = np.zeros((17, 3), dtype=np.float32)
    # Fill with defaults at center
    for i in range(17):
        kps[i] = [320, 240, default_conf]
    # Override specific keypoints
    kps[0] = nose
    kps[5] = left_shoulder
    kps[6] = right_shoulder
    kps[11] = left_hip
    kps[12] = right_hip
    return kps


def make_standing_keypoints() -> np.ndarray:
    """Standing person: nose above hips, vertically aligned."""
    return make_keypoints(
        nose=(320, 50, 0.95),
        left_shoulder=(290, 130, 0.9),
        right_shoulder=(350, 130, 0.9),
        left_hip=(295, 280, 0.9),
        right_hip=(345, 280, 0.9),
    )


def make_fallen_keypoints() -> np.ndarray:
    """Fallen person: lying horizontal, nose and hips at similar Y."""
    return make_keypoints(
        nose=(150, 400, 0.9),
        left_shoulder=(250, 400, 0.9),
        right_shoulder=(350, 400, 0.9),
        left_hip=(400, 410, 0.9),
        right_hip=(500, 410, 0.9),
    )


def make_low_confidence_keypoints() -> np.ndarray:
    """All keypoints have very low confidence."""
    kps = make_standing_keypoints()
    kps[:, 2] = 0.1
    return kps


# --- FeatureVector Tests ---

class TestFeatureVector:
    def test_to_array(self):
        fv = FeatureVector(0.1, 0.5, -0.02, 0.4, 0.85)
        arr = fv.to_array()
        assert arr.shape == (NUM_FEATURES,)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, [0.1, 0.5, -0.02, 0.4, 0.85], rtol=1e-5)

    def test_from_array(self):
        arr = np.array([0.1, 0.5, -0.02, 0.4, 0.85], dtype=np.float32)
        fv = FeatureVector.from_array(arr)
        assert fv.body_angle == pytest.approx(0.1, abs=1e-5)
        assert fv.cog_height == pytest.approx(0.5, abs=1e-5)

    def test_roundtrip(self):
        original = FeatureVector(0.123, 0.456, -0.078, 1.234, 0.567)
        restored = FeatureVector.from_array(original.to_array())
        assert restored.body_angle == pytest.approx(original.body_angle, abs=1e-5)
        assert restored.cog_height == pytest.approx(original.cog_height, abs=1e-5)


# --- FeatureExtractor Tests ---

class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_standing_person(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)

        # Standing: body angle should be small (near vertical)
        assert fv.body_angle < 0.2  # < 36 degrees normalized
        # CoG (hips) at y≈280 in 480px frame
        assert 0.5 < fv.cog_height < 0.7
        # First frame, no velocity
        assert fv.vertical_velocity == 0.0
        # Standing bbox is tall/narrow (ratio < 1)
        assert fv.bbox_aspect_ratio < 0.5
        # Good confidence
        assert fv.keypoint_confidence > 0.8

    def test_fallen_person(self):
        kps = make_fallen_keypoints()
        bbox = (100.0, 380.0, 550.0, 430.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)

        # Fallen: body angle should be large (near horizontal)
        assert fv.body_angle > 0.3  # > 54 degrees normalized
        # CoG (hips) near bottom of frame
        assert fv.cog_height > 0.8
        # Fallen bbox is wide/short (ratio > 1)
        assert fv.bbox_aspect_ratio > 1.0

    def test_velocity_computation(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)

        # First frame
        fv1 = self.extractor.extract(kps, bbox, frame_height=480, prev_cog_height=None)
        assert fv1.vertical_velocity == 0.0

        # Second frame with known previous CoG
        fv2 = self.extractor.extract(kps, bbox, frame_height=480, prev_cog_height=0.3)
        # CoG should be around 0.58, velocity = 0.58 - 0.3 ≈ 0.28
        assert fv2.vertical_velocity > 0.0

    def test_missing_keypoints(self):
        kps = make_low_confidence_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)

        # Should not crash, should return valid features
        assert not math.isnan(fv.body_angle)
        assert not math.isnan(fv.cog_height)
        assert not math.isnan(fv.vertical_velocity)
        assert not math.isnan(fv.bbox_aspect_ratio)
        assert fv.keypoint_confidence < 0.2

    def test_zero_height_bbox(self):
        kps = make_standing_keypoints()
        bbox = (100.0, 200.0, 300.0, 200.0)  # height = 0
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        assert fv.bbox_aspect_ratio == 0.0

    def test_zero_frame_height(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=0)
        assert fv.cog_height == 0.5  # fallback

    def test_velocity_clamping(self):
        kps = make_standing_keypoints()
        bbox = (270.0, 40.0, 360.0, 450.0)
        # Extreme previous CoG to test clamping
        fv = self.extractor.extract(kps, bbox, frame_height=480, prev_cog_height=-5.0)
        assert -1.0 <= fv.vertical_velocity <= 1.0

    def test_no_nan_or_inf(self):
        """Feature values should never be NaN or Inf."""
        test_cases = [
            (make_standing_keypoints(), (270, 40, 360, 450)),
            (make_fallen_keypoints(), (100, 380, 550, 430)),
            (make_low_confidence_keypoints(), (270, 40, 360, 450)),
            (np.zeros((17, 3), dtype=np.float32), (0, 0, 0, 0)),
        ]
        for kps, bbox in test_cases:
            fv = self.extractor.extract(kps, bbox, frame_height=480)
            arr = fv.to_array()
            assert not np.any(np.isnan(arr)), f"NaN found in features: {arr}"
            assert not np.any(np.isinf(arr)), f"Inf found in features: {arr}"

    def test_nan_keypoint_coordinates(self):
        """NaN coordinates with high confidence should not produce NaN features."""
        kps = make_standing_keypoints()
        kps[0, 0] = float("nan")  # Corrupt nose X
        kps[0, 1] = float("nan")  # Corrupt nose Y
        kps[0, 2] = 0.95  # High confidence — the dangerous case
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr)), f"NaN propagated to features: {arr}"
        assert not np.any(np.isinf(arr)), f"Inf propagated to features: {arr}"

    def test_nan_hip_coordinates(self):
        """NaN hip coordinates should not corrupt CoG height."""
        kps = make_standing_keypoints()
        kps[11, 0:2] = float("nan")  # Corrupt left hip
        kps[11, 2] = 0.9
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        arr = fv.to_array()
        assert not np.any(np.isnan(arr)), f"NaN propagated from hip: {arr}"

    def test_single_hip_visible(self):
        kps = make_standing_keypoints()
        kps[12, 2] = 0.0  # Hide right hip
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        assert not math.isnan(fv.body_angle)
        assert not math.isnan(fv.cog_height)

    def test_shoulder_fallback(self):
        """When hips are not visible, should use shoulders."""
        kps = make_standing_keypoints()
        kps[11, 2] = 0.0  # Hide left hip
        kps[12, 2] = 0.0  # Hide right hip
        bbox = (270.0, 40.0, 360.0, 450.0)
        fv = self.extractor.extract(kps, bbox, frame_height=480)
        # Should still compute a valid body angle using shoulders
        assert not math.isnan(fv.body_angle)


# --- TrackState Tests ---

class TestTrackState:
    def test_creation(self):
        state = TrackState(track_id=1, window_size=30)
        assert state.track_id == 1
        assert state.prev_cog_height is None
        assert state.total_frames_seen == 0
        assert not state.is_ready()

    def test_add_features(self):
        state = TrackState(track_id=1, window_size=5)
        for i in range(5):
            fv = FeatureVector(0.1, 0.5 + i * 0.01, 0.0, 0.4, 0.9)
            state.add_features(fv, frame_num=i)

        assert state.total_frames_seen == 5
        assert state.is_ready()
        assert state.buffer_fill == 1.0
        assert state.prev_cog_height == pytest.approx(0.54)

    def test_sliding_window(self):
        state = TrackState(track_id=1, window_size=3)
        for i in range(5):
            fv = FeatureVector(float(i), 0.5, 0.0, 0.4, 0.9)
            state.add_features(fv, frame_num=i)

        seq = state.get_sequence()
        assert seq is not None
        assert seq.shape == (3, NUM_FEATURES)
        # Should contain last 3 values: 2, 3, 4
        assert seq[0, 0] == pytest.approx(2.0)
        assert seq[2, 0] == pytest.approx(4.0)

    def test_get_sequence_not_ready(self):
        state = TrackState(track_id=1, window_size=30)
        state.add_features(FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9), 0)
        assert state.get_sequence() is None


# --- TrackManager Tests ---

class TestTrackManager:
    def test_create_and_update(self):
        manager = TrackManager(window_size=5)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)
        manager.update(track_id=1, features=fv, frame_num=0)

        assert manager.active_count == 1
        assert manager.total_tracks_created == 1

    def test_skip_negative_track_id(self):
        manager = TrackManager(window_size=5)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)
        manager.update(track_id=-1, features=fv, frame_num=0)
        assert manager.active_count == 0

    def test_multiple_tracks(self):
        manager = TrackManager(window_size=5)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)
        manager.update(track_id=1, features=fv, frame_num=0)
        manager.update(track_id=2, features=fv, frame_num=0)
        manager.update(track_id=3, features=fv, frame_num=0)

        assert manager.active_count == 3
        assert manager.total_tracks_created == 3

    def test_get_ready_tracks(self):
        manager = TrackManager(window_size=3)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)

        # Track 1: fill buffer
        for i in range(3):
            manager.update(1, fv, i)
        # Track 2: only 1 frame
        manager.update(2, fv, 0)

        ready = manager.get_ready_tracks()
        assert ready == [1]

    def test_cleanup_stale_tracks(self):
        manager = TrackManager(window_size=5, stale_threshold=10)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)

        # Track seen at frame 0
        manager.update(1, fv, frame_num=0)
        # Track seen at frame 50
        manager.update(2, fv, frame_num=50)

        removed = manager.cleanup(current_frame=55)
        assert 1 in removed  # 55 - 0 > 10
        assert 2 not in removed  # 55 - 50 = 5 < 10
        assert manager.active_count == 1

    def test_get_prev_cog_height(self):
        manager = TrackManager(window_size=5)
        fv = FeatureVector(0.1, 0.65, 0.0, 0.4, 0.9)
        manager.update(1, fv, 0)

        assert manager.get_prev_cog_height(1) == pytest.approx(0.65)
        assert manager.get_prev_cog_height(999) is None

    def test_classification(self):
        manager = TrackManager(window_size=5)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)
        manager.update(1, fv, 0)

        manager.set_classification(1, "falling", 0.92)
        label, conf = manager.get_classification(1)
        assert label == "falling"
        assert conf == pytest.approx(0.92)

    def test_classification_unknown_track(self):
        manager = TrackManager(window_size=5)
        label, conf = manager.get_classification(999)
        assert label == "normal"
        assert conf == 0.0

    def test_cleanup_exact_boundary(self):
        """Track at exactly stale_threshold should NOT be removed."""
        manager = TrackManager(window_size=5, stale_threshold=10)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)
        manager.update(1, fv, frame_num=0)

        removed = manager.cleanup(current_frame=10)  # 10 - 0 = 10, NOT > 10
        assert 1 not in removed
        assert manager.active_count == 1

    def test_reset(self):
        manager = TrackManager(window_size=5)
        fv = FeatureVector(0.1, 0.5, 0.0, 0.4, 0.9)
        manager.update(1, fv, 0)
        manager.update(2, fv, 0)

        manager.reset()
        assert manager.active_count == 0
        assert manager.total_tracks_created == 0
