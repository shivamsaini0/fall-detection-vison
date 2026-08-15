"""Tests for the person detector module."""

import numpy as np

from src.detector import DetectedPerson, FrameResult
from src.utils import draw_alert, draw_bbox, draw_skeleton, draw_status_bar


class TestDetectedPerson:
    def test_bbox_properties(self):
        person = DetectedPerson(
            track_id=1,
            bbox=(100.0, 200.0, 300.0, 500.0),
            keypoints=np.zeros((17, 3)),
            confidence=0.9,
        )
        assert person.bbox_width == 200.0
        assert person.bbox_height == 300.0
        assert person.bbox_center == (200.0, 350.0)

    def test_zero_size_bbox(self):
        person = DetectedPerson(
            track_id=1,
            bbox=(100.0, 100.0, 100.0, 100.0),
            keypoints=np.zeros((17, 3)),
            confidence=0.5,
        )
        assert person.bbox_width == 0.0
        assert person.bbox_height == 0.0


class TestFrameResult:
    def test_empty(self):
        result = FrameResult(frame_num=0)
        assert result.num_persons == 0
        assert result.track_ids == []

    def test_with_persons(self):
        persons = [
            DetectedPerson(1, (0, 0, 10, 10), np.zeros((17, 3)), 0.9),
            DetectedPerson(2, (20, 20, 30, 30), np.zeros((17, 3)), 0.8),
        ]
        result = FrameResult(frame_num=5, persons=persons)
        assert result.num_persons == 2
        assert result.track_ids == [1, 2]


class TestDrawSkeleton:
    def test_draw_on_blank_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Place some visible keypoints
        keypoints[0] = [320, 100, 0.9]  # nose
        keypoints[5] = [280, 200, 0.9]  # left shoulder
        keypoints[6] = [360, 200, 0.9]  # right shoulder
        keypoints[11] = [280, 350, 0.9]  # left hip
        keypoints[12] = [360, 350, 0.9]  # right hip

        result = draw_skeleton(frame, keypoints)
        # Frame should have non-zero pixels where skeleton is drawn
        assert np.any(result > 0)

    def test_draw_with_low_confidence(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # All keypoints have low confidence
        keypoints[:, 2] = 0.1
        keypoints[:, 0] = 320
        keypoints[:, 1] = 240

        result = draw_skeleton(frame, keypoints, confidence_threshold=0.5)
        # Nothing should be drawn
        assert not np.any(result > 0)

    def test_draw_with_none_keypoints(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_skeleton(frame, None)
        assert not np.any(result > 0)

    def test_draw_with_empty_keypoints(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_skeleton(frame, np.array([]))
        assert not np.any(result > 0)


class TestDrawBbox:
    def test_draw_simple_bbox(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bbox(frame, (100, 100, 300, 400))
        assert np.any(frame > 0)

    def test_draw_with_label(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bbox(frame, (100, 50, 300, 400), label="ID:1 0.95")
        assert np.any(frame > 0)

    def test_label_near_top_of_frame(self):
        """Label should draw below bbox when too close to top edge."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bbox(frame, (100, 5, 300, 200), label="ID:1")
        assert np.any(frame > 0)


class TestDrawAlert:
    def test_draw_centered_alert(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_alert(frame, "FALL DETECTED")
        assert np.any(result > 0)

    def test_draw_alert_at_position(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_alert(frame, "WARNING", position=(50, 100))
        assert np.any(result > 0)


class TestDrawStatusBar:
    def test_draw_status_bar(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_status_bar(frame, 100, 1000, 30.0, 2, 1)
        # Bottom of frame should have content
        assert np.any(result[450:, :] > 0)

    def test_zero_total_frames(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_status_bar(frame, 0, 0, 30.0, 0, 0)
        assert result.shape == (480, 640, 3)
