"""Lightweight IoU-based multi-object tracker for edge deployment.

No PyTorch or Ultralytics dependency — uses only numpy and scipy.
Designed to replace BoT-SORT when running on edge devices with ONNX Runtime.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class SimpleTracker:
    """IoU-based multi-object tracker using Hungarian algorithm.

    Assigns persistent track IDs to bounding box detections across frames.
    Uses Intersection-over-Union (IoU) for matching detections to existing tracks.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30):
        """Initialize tracker.

        Args:
            iou_threshold: Minimum IoU to match a detection to a track.
            max_lost: Remove tracks not seen for this many frames.
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 1
        self._tracks: dict[int, dict] = {}  # id → {bbox, lost_count}

    def update(self, bboxes: np.ndarray) -> list[int]:
        """Update tracker with new detections and return track IDs.

        Args:
            bboxes: Array of shape (N, 4) with [x1, y1, x2, y2] per detection.

        Returns:
            List of track IDs (length N), one per input bbox.
        """
        if len(bboxes) == 0:
            self._increment_lost()
            self._remove_stale()
            return []

        bboxes = np.array(bboxes, dtype=np.float32)

        if not self._tracks:
            # First frame — create new tracks for all detections
            track_ids = []
            for bbox in bboxes:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": bbox, "lost_count": 0}
                track_ids.append(tid)
            return track_ids

        # Compute IoU matrix between existing tracks and new detections
        track_ids_list = list(self._tracks.keys())
        track_bboxes = np.array([self._tracks[tid]["bbox"] for tid in track_ids_list])
        iou_matrix = self._compute_iou_matrix(track_bboxes, bboxes)

        # Hungarian algorithm for optimal matching
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Assign matches
        assigned_track_ids = [-1] * len(bboxes)
        matched_tracks = set()
        matched_detections = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                tid = track_ids_list[row]
                assigned_track_ids[col] = tid
                self._tracks[tid]["bbox"] = bboxes[col]
                self._tracks[tid]["lost_count"] = 0
                matched_tracks.add(tid)
                matched_detections.add(col)

        # Create new tracks for unmatched detections
        for i in range(len(bboxes)):
            if i not in matched_detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": bboxes[i], "lost_count": 0}
                assigned_track_ids[i] = tid

        # Increment lost count for unmatched tracks
        for tid in track_ids_list:
            if tid not in matched_tracks:
                self._tracks[tid]["lost_count"] += 1

        self._remove_stale()

        return assigned_track_ids

    def _increment_lost(self):
        """Increment lost count for all tracks (no detections this frame)."""
        for tid in self._tracks:
            self._tracks[tid]["lost_count"] += 1

    def _remove_stale(self):
        """Remove tracks not seen for max_lost frames."""
        stale = [tid for tid, t in self._tracks.items() if t["lost_count"] > self.max_lost]
        for tid in stale:
            del self._tracks[tid]

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of bounding boxes.

        Args:
            boxes_a: (M, 4) array of [x1, y1, x2, y2].
            boxes_b: (N, 4) array of [x1, y1, x2, y2].

        Returns:
            (M, N) IoU matrix.
        """
        m, n = len(boxes_a), len(boxes_b)
        iou = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            for j in range(n):
                x1 = max(boxes_a[i, 0], boxes_b[j, 0])
                y1 = max(boxes_a[i, 1], boxes_b[j, 1])
                x2 = min(boxes_a[i, 2], boxes_b[j, 2])
                y2 = min(boxes_a[i, 3], boxes_b[j, 3])

                inter = max(0, x2 - x1) * max(0, y2 - y1)

                area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
                area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (boxes_b[j, 3] - boxes_b[j, 1])

                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0.0

        return iou

    def reset(self):
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return len(self._tracks)
