#!/usr/bin/env python3
"""Lightweight edge inference for fall detection — NO PyTorch required.

Uses only: onnxruntime + opencv + numpy + scipy
Total runtime dependencies: ~150MB (vs ~2.5GB with PyTorch)

Usage:
    python scripts/edge_inference.py \
        --source tests/test_videos/elderly_falling.mp4 --output output/edge_result.mp4
    python scripts/edge_inference.py --source 0  # live camera
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction import FeatureExtractor
from src.simple_tracker import SimpleTracker
from src.utils import (
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    draw_alert,
    draw_skeleton,
    draw_status_bar,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "normal", 1: "falling", 2: "fallen"}
STATE_COLORS = {"normal": COLOR_GREEN, "falling": COLOR_RED, "fallen": COLOR_ORANGE}


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def letterbox(frame, target_size=640):
    """Resize frame with letterbox padding (maintain aspect ratio)."""
    h, w = frame.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    left = pad_w // 2

    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, top, left


def parse_yolo_output(output, conf_thresh, orig_h, orig_w, scale, pad_top, pad_left):
    """Parse raw YOLO ONNX output into bboxes and keypoints.

    YOLO pose output shape: (1, 56, num_anchors)
    - 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints x,y,conf)
    """
    # Output shape: (1, 56, num_anchors) → transpose to (num_anchors, 56)
    preds = output[0][0].T  # (8400, 56)

    # Filter by confidence
    confidences = preds[:, 4]
    mask = confidences >= conf_thresh
    preds = preds[mask]

    if len(preds) == 0:
        return [], []

    # Extract components
    bboxes_xywh = preds[:, :4]  # center_x, center_y, w, h
    confs = preds[:, 4]
    kps_raw = preds[:, 5:]  # 17*3 = 51 values

    # Convert xywh to xyxy
    bboxes = np.zeros_like(bboxes_xywh)
    bboxes[:, 0] = bboxes_xywh[:, 0] - bboxes_xywh[:, 2] / 2  # x1
    bboxes[:, 1] = bboxes_xywh[:, 1] - bboxes_xywh[:, 3] / 2  # y1
    bboxes[:, 2] = bboxes_xywh[:, 0] + bboxes_xywh[:, 2] / 2  # x2
    bboxes[:, 3] = bboxes_xywh[:, 1] + bboxes_xywh[:, 3] / 2  # y2

    # NMS
    indices = _nms(bboxes, confs, iou_thresh=0.45)
    bboxes = bboxes[indices]
    confs = confs[indices]
    kps_raw = kps_raw[indices]

    # Rescale coordinates from 640x640 back to original
    results_bboxes = []
    results_keypoints = []

    for i in range(len(bboxes)):
        # Remove padding offset and scale back
        bbox = bboxes[i].copy()
        bbox[0] = (bbox[0] - pad_left) / scale
        bbox[1] = (bbox[1] - pad_top) / scale
        bbox[2] = (bbox[2] - pad_left) / scale
        bbox[3] = (bbox[3] - pad_top) / scale

        # Clamp to frame bounds
        bbox[0] = max(0, min(bbox[0], orig_w))
        bbox[1] = max(0, min(bbox[1], orig_h))
        bbox[2] = max(0, min(bbox[2], orig_w))
        bbox[3] = max(0, min(bbox[3], orig_h))

        results_bboxes.append(bbox)

        # Parse keypoints (17 x [x, y, conf])
        kps = kps_raw[i].reshape(17, 3).copy()
        kps[:, 0] = (kps[:, 0] - pad_left) / scale
        kps[:, 1] = (kps[:, 1] - pad_top) / scale
        results_keypoints.append(kps)

    return results_bboxes, results_keypoints


def _nms(bboxes, scores, iou_thresh=0.45):
    """Simple NMS implementation."""
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]

    return keep


class EdgeFallDetector:
    """Complete fall detection pipeline using ONNX Runtime only."""

    def __init__(
        self,
        yolo_path="models/exports/yolo11n-pose.onnx",
        lstm_path="models/exports/fall_detector.onnx",
        conf_threshold=0.40,
        num_threads=4,
    ):
        sess_opts = rt.SessionOptions()
        sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.inter_op_num_threads = num_threads
        sess_opts.intra_op_num_threads = num_threads
        providers = ["CPUExecutionProvider"]

        logger.info(f"Loading YOLO ONNX: {yolo_path}")
        self.yolo = rt.InferenceSession(yolo_path, sess_opts, providers)
        self.yolo_input_name = self.yolo.get_inputs()[0].name

        logger.info(f"Loading LSTM ONNX: {lstm_path}")
        self.lstm = rt.InferenceSession(lstm_path, sess_opts, providers)
        self.lstm_input_name = self.lstm.get_inputs()[0].name

        self.tracker = SimpleTracker(iou_threshold=0.3, max_lost=30)
        self.feature_extractor = FeatureExtractor()
        self.conf_threshold = conf_threshold

        # Per-track state
        self.track_buffers: dict[int, deque] = {}
        self.track_prev_cog: dict[int, float] = {}
        self.track_states: dict[int, tuple[str, float]] = {}
        self.fall_sticky: dict[int, bool] = {}

    def process_video(self, source, output_path=None, json_path=None):
        """Process video/camera source for fall detection."""
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise IOError(f"Cannot open source: {source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale_vis = max(h / 720, 1.0)

        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        logger.info(f"Processing: {w}x{h} @ {fps:.0f}fps, {total} frames")

        events = []
        unique_tracks = set()
        falls_detected = 0
        start_time = time.time()

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. YOLO inference
            img, sc, pt, pl = letterbox(frame)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = img_rgb.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis]

            yolo_out = self.yolo.run(None, {self.yolo_input_name: blob})
            bboxes, keypoints_list = parse_yolo_output(
                yolo_out, 0.3, h, w, sc, pt, pl
            )

            # 2. Track
            if bboxes:
                bbox_array = np.array(bboxes)
                track_ids = self.tracker.update(bbox_array)
            else:
                track_ids = self.tracker.update(np.empty((0, 4)))

            # 3. Feature extraction + LSTM classification per person
            active_ids = set()
            for i, tid in enumerate(track_ids):
                unique_tracks.add(tid)
                active_ids.add(tid)
                bbox = bboxes[i]
                kps = keypoints_list[i]

                prev_cog = self.track_prev_cog.get(tid)
                features = self.feature_extractor.extract(kps, tuple(bbox), h, prev_cog)
                self.track_prev_cog[tid] = features.cog_height

                if tid not in self.track_buffers:
                    self.track_buffers[tid] = deque(maxlen=30)
                self.track_buffers[tid].append(features.to_array())

                state, conf = "normal", 0.0
                if len(self.track_buffers[tid]) >= 30:
                    seq = np.array(list(self.track_buffers[tid]), dtype=np.float32)
                    lstm_in = seq[np.newaxis]
                    logits = self.lstm.run(None, {self.lstm_input_name: lstm_in})[0]
                    probs = softmax(logits)[0]
                    cls = int(np.argmax(probs))
                    conf = float(probs[cls])
                    state = CLASS_NAMES[cls]

                # Sticky fall
                if state in ("falling", "fallen") and conf >= self.conf_threshold:
                    self.fall_sticky[tid] = True
                    if not events or events[-1].get("track_id") != tid:
                        events.append({
                            "type": "fall", "track_id": tid,
                            "frame": frame_num, "confidence": round(conf, 3),
                        })
                        falls_detected = len(events)
                elif tid in self.fall_sticky and self.fall_sticky[tid]:
                    is_standing = (
                        features.body_angle < 0.15
                        and features.cog_height < 0.55
                        and features.bbox_aspect_ratio < 0.8
                    )
                    if is_standing:
                        self.fall_sticky[tid] = False
                    else:
                        state = "fallen"
                        conf = max(conf, 0.5)

                self.track_states[tid] = (state, conf)

                # Draw
                color = STATE_COLORS.get(state, COLOR_GREEN)
                draw_skeleton(frame, kps, color=color,
                              thickness=max(2, int(2 * scale_vis)),
                              point_radius=max(4, int(4 * scale_vis)))

                # Label
                label = f"ID:{tid} {state.upper()}"
                if state != "normal":
                    label += f" {conf:.0%}"
                _draw_label(frame, bbox, label, color, state, scale_vis)

            # Alerts
            for tid, (st, co) in self.track_states.items():
                if tid in active_ids and st == "falling" and co >= self.conf_threshold:
                    draw_alert(frame, "!! FALL DETECTED !!", color=COLOR_RED,
                               font_scale=1.2 * scale_vis, thickness=max(3, int(3 * scale_vis)))
                    break
                elif tid in active_ids and st == "fallen" and co >= self.conf_threshold:
                    draw_alert(frame, "PERSON FALLEN", color=COLOR_ORANGE,
                               font_scale=1.2 * scale_vis, thickness=max(3, int(3 * scale_vis)))
                    break

            draw_status_bar(frame, frame_num, total, fps, len(active_ids), falls_detected)

            if writer:
                writer.write(frame)
            frame_num += 1

        cap.release()
        if writer:
            writer.release()
            # Convert to H.264
            _convert_h264(output_path)

        proc_time = time.time() - start_time

        result = {
            "source": str(source),
            "frames_processed": frame_num,
            "falls_detected": len(events),
            "persons_tracked": len(unique_tracks),
            "processing_time": round(proc_time, 1),
            "fps": round(frame_num / proc_time, 1) if proc_time > 0 else 0,
            "events": events,
            "runtime": "ONNX Runtime (CPU)",
        }

        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

        logger.info(f"Done: {frame_num} frames, {len(events)} falls, "
                     f"{result['fps']} FPS, {proc_time:.1f}s")
        return result


def _draw_label(frame, bbox, label, color, state, scale):
    """Draw scaled label on frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    thick = max(2, int(3 * scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.7 * scale
    ft = max(1, int(2 * scale))
    pad = int(6 * scale)
    (tw, th), bl = cv2.getTextSize(label, font, fs, ft)

    ly1 = y1 - th - bl - pad * 2 if y1 - th - pad * 2 >= 0 else y1
    ly2 = y1 if y1 - th - pad * 2 >= 0 else y1 + th + bl + pad * 2

    cv2.rectangle(frame, (x1, ly1), (x1 + tw + pad * 2, ly2), color, -1)
    text_color = (255, 255, 255) if state != "normal" else (0, 0, 0)
    ty = ly2 - bl - pad if y1 - th - pad * 2 >= 0 else ly1 + th + pad
    cv2.putText(frame, label, (x1 + pad, ty), font, fs, text_color, ft, cv2.LINE_AA)


def _convert_h264(path):
    """Convert to H.264 using ffmpeg if available."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
             "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(tmp)],
            capture_output=True, timeout=300,
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description="Edge fall detection (ONNX Runtime)")
    parser.add_argument("--source", "-s", required=True, help="Video file or camera index (0)")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument("--json", "-j", default=None, help="JSON output path")
    parser.add_argument("--yolo", default="models/exports/yolo11n-pose.onnx")
    parser.add_argument("--lstm", default="models/exports/fall_detector.onnx")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    detector = EdgeFallDetector(
        yolo_path=args.yolo, lstm_path=args.lstm,
        conf_threshold=args.confidence, num_threads=args.threads,
    )
    detector.process_video(source, args.output, args.json)


if __name__ == "__main__":
    main()
