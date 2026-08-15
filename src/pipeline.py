"""End-to-end fall detection pipeline: video → annotated output + JSON log."""

import json
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.fall_detector import FallDetector
from src.detector import PersonDetector
from src.feature_extraction import FeatureExtractor
from src.track_manager import TrackManager
from src.utils import (
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    draw_alert,
    draw_skeleton,
    draw_status_bar,
)
from src.video_io import VideoReader, VideoWriter, format_timestamp

logger = logging.getLogger(__name__)

# State colors
STATE_COLORS = {
    "normal": COLOR_GREEN,
    "falling": COLOR_RED,
    "fallen": COLOR_ORANGE,
}


class FallEvent:
    """Represents a detected fall event."""

    def __init__(self, track_id: int, start_frame: int, fps: float, confidence: float):
        self.track_id = track_id
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.fps = fps
        self.confidence = confidence
        self.state = "falling"

    def update(self, frame_num: int, confidence: float, state: str):
        self.end_frame = frame_num
        self.confidence = max(self.confidence, confidence)
        self.state = state

    def to_dict(self) -> dict:
        return {
            "type": "fall",
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "timestamp_start": format_timestamp(self.start_frame, self.fps),
            "timestamp_end": format_timestamp(self.end_frame, self.fps),
            "confidence": round(self.confidence, 3),
        }


class FallDetectionPipeline:
    """Full pipeline: video → YOLO detection → tracking → features → LSTM → output."""

    def __init__(
        self,
        yolo_model: str = "yolo11n-pose.pt",
        lstm_checkpoint: str = "models/checkpoints/best.pth",
        device: str | None = None,
        confidence_threshold: float = 0.40,
        cooldown_frames: int = 90,
        tracker_config: str = "configs/botsort.yaml",
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing pipeline on {self.device}...")

        # YOLO detector
        self.detector = PersonDetector(
            model_name=yolo_model,
            device=self.device,
            confidence_threshold=0.3,
            tracker_config=tracker_config,
        )

        # LSTM classifier
        self.lstm = FallDetector()
        ckpt = torch.load(lstm_checkpoint, map_location=self.device, weights_only=False)
        self.lstm.load_state_dict(ckpt["model_state_dict"])
        self.lstm.to(self.device)
        self.lstm.eval()
        logger.info(f"LSTM loaded from {lstm_checkpoint}")

        # Feature extraction
        self.feature_extractor = FeatureExtractor()
        self.track_manager = TrackManager(window_size=30, stale_threshold=90)

        # Settings
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames

        # State
        self._interrupted = False
        self._cooldowns: dict[int, int] = {}  # track_id → last alert frame
        self._active_events: dict[int, FallEvent] = {}
        self._all_events: list[FallEvent] = []
        self._fall_sticky: dict[int, int] = {}  # track_id → frame when fall detected
        # Remember fallen person's last bbox for occlusion persistence
        self._fallen_positions: dict[int, tuple] = {}  # track_id → (bbox, last_frame)

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> dict:
        """Process a video file for fall detection.

        Args:
            input_path: Path to input video.
            output_path: Path for annotated output video (None to skip).
            json_path: Path for JSON event log (None to skip).

        Returns:
            Dict with events, summary, and metadata.
        """
        input_path = Path(input_path)
        self._interrupted = False
        self._cooldowns.clear()
        self._active_events.clear()
        self._all_events.clear()
        self._fall_sticky.clear()
        self._fallen_positions.clear()
        self.track_manager.reset()
        self.detector.reset_tracker()

        # Setup signal handler for graceful Ctrl+C
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        start_time = time.time()
        falls_detected = 0
        unique_tracks = set()

        try:
            with VideoReader(input_path) as reader:
                meta = reader.metadata
                writer = None

                if output_path:
                    output_path = Path(output_path)
                    writer = VideoWriter(output_path, meta.width, meta.height, meta.fps)
                    writer.__enter__()

                logger.info(
                    f"Processing {input_path.name}: "
                    f"{meta.width}x{meta.height} @ {meta.fps:.1f}fps, "
                    f"{meta.total_frames} frames ({meta.duration_seconds:.1f}s)"
                )

                for frame_num, frame in tqdm(
                    reader, total=meta.total_frames, desc="Processing"
                ):
                    if self._interrupted:
                        logger.info("Interrupted by user, saving partial output...")
                        break

                    # 1. Detect + Track
                    frame_result = self.detector.detect_frame(frame, frame_num)

                    # 2. Extract features + classify per person
                    active_ids = set()
                    for person in frame_result.persons:
                        if person.track_id < 0:
                            continue

                        active_ids.add(person.track_id)
                        unique_tracks.add(person.track_id)

                        # Extract features
                        prev_cog = self.track_manager.get_prev_cog_height(person.track_id)
                        features = self.feature_extractor.extract(
                            person.keypoints, person.bbox, meta.height, prev_cog
                        )
                        self.track_manager.update(person.track_id, features, frame_num)

                        # Classify if buffer is full
                        seq = self.track_manager.get_sequence(person.track_id)
                        state = "normal"
                        conf = 0.0

                        if seq is not None:
                            state, conf = self._classify(seq)

                            # Sticky fall: once fallen, stay "fallen" until
                            # person clearly stands (upright angle + low CoG)
                            tid = person.track_id
                            if state in ("falling", "fallen") and conf >= self.confidence_threshold:
                                self._fall_sticky[tid] = frame_num
                            elif tid in self._fall_sticky:
                                is_standing = (
                                    features.body_angle < 0.15
                                    and features.cog_height < 0.55
                                    and features.bbox_aspect_ratio < 0.8
                                )
                                if is_standing:
                                    del self._fall_sticky[tid]
                                else:
                                    state = "fallen"
                                    conf = max(conf, 0.5)

                            self.track_manager.set_classification(person.track_id, state, conf)

                        # Handle fall events
                        if state in ("falling", "fallen") and conf >= self.confidence_threshold:
                            self._handle_fall_event(
                                person.track_id, frame_num, state, conf, meta.fps
                            )
                            falls_detected = len(self._all_events)

                        # Draw on frame — scale sizes to resolution
                        scale = max(meta.height / 720, 1.0)

                        # Determine display state and color
                        if state in ("falling", "fallen") and conf >= self.confidence_threshold:
                            display_state = state
                        else:
                            display_state = "normal"

                        color = STATE_COLORS.get(display_state, COLOR_GREEN)

                        # Build label — always show class with confidence
                        state_labels = {
                            "normal": "NORMAL",
                            "falling": "FALLING",
                            "fallen": "FALLEN",
                        }
                        if display_state != "normal":
                            label = (
                                f"ID:{person.track_id} "
                                f"{state_labels[display_state]} {conf:.0%}"
                            )
                        else:
                            label = f"ID:{person.track_id} {state_labels[display_state]}"

                        draw_skeleton(
                            frame, person.keypoints, color=color,
                            thickness=max(2, int(2 * scale)),
                            point_radius=max(4, int(4 * scale)),
                        )
                        self._draw_label_scaled(
                            frame, person.bbox, label, color, state, scale
                        )

                        # Save position of fallen persons for occlusion persistence
                        if display_state in ("falling", "fallen"):
                            self._fallen_positions[person.track_id] = (
                                person.bbox, frame_num
                            )

                    # Draw persistent fallen markers for occluded tracks
                    scale = max(meta.height / 720, 1.0)
                    for tid, (bbox, last_frame) in list(self._fallen_positions.items()):
                        if tid in active_ids:
                            continue  # Still tracked, already drawn above
                        frames_since = frame_num - last_frame
                        if frames_since > meta.fps * 10:
                            # Lost for >10 seconds, remove
                            del self._fallen_positions[tid]
                            continue
                        # Draw ghost bbox for occluded fallen person
                        self._draw_label_scaled(
                            frame, bbox,
                            f"ID:{tid} FALLEN (occluded)",
                            COLOR_ORANGE, "fallen", scale,
                        )

                    # Draw alerts based on current per-person state
                    alert_scale = max(meta.height / 720, 1.0)
                    has_falling = False
                    has_fallen = False
                    fallen_duration = 0.0

                    for person in frame_result.persons:
                        tid = person.track_id
                        t_state, t_conf = self.track_manager.get_classification(tid)
                        if t_state == "falling" and t_conf >= self.confidence_threshold:
                            has_falling = True
                        elif t_state == "fallen" and t_conf >= self.confidence_threshold:
                            has_fallen = True
                            if tid in self._active_events:
                                fallen_duration = max(
                                    fallen_duration,
                                    (frame_num - self._active_events[tid].start_frame)
                                    / max(meta.fps, 1),
                                )

                    # Also check occluded fallen persons
                    for tid in self._fallen_positions:
                        if tid not in active_ids and tid in self._active_events:
                            has_fallen = True
                            fallen_duration = max(
                                fallen_duration,
                                (frame_num - self._active_events[tid].start_frame)
                                / max(meta.fps, 1),
                            )

                    if has_falling:
                        draw_alert(
                            frame, "!! FALL DETECTED !!",
                            color=COLOR_RED,
                            font_scale=1.2 * alert_scale,
                            thickness=max(3, int(3 * alert_scale)),
                        )
                    elif has_fallen:
                        draw_alert(
                            frame,
                            f"PERSON FALLEN ({fallen_duration:.1f}s)",
                            color=COLOR_ORANGE,
                            font_scale=1.2 * alert_scale,
                            thickness=max(3, int(3 * alert_scale)),
                        )

                    # Status bar
                    draw_status_bar(
                        frame, frame_num, meta.total_frames,
                        meta.fps, len(active_ids), falls_detected,
                    )

                    # Write annotated frame
                    if writer:
                        writer.write(frame)

                    # Periodic cleanup
                    if frame_num % 100 == 0:
                        self.track_manager.cleanup(frame_num)

                if writer:
                    writer.__exit__(None, None, None)

        finally:
            signal.signal(signal.SIGINT, original_handler)

        processing_time = time.time() - start_time

        # Build result
        result = {
            "metadata": {
                "input_file": str(input_path),
                "processing_date": datetime.now().isoformat(),
                "model_version": "1.0",
                "confidence_threshold": self.confidence_threshold,
                "video_fps": meta.fps,
                "video_duration": format_timestamp(meta.total_frames, meta.fps),
                "total_frames": meta.total_frames,
                "resolution": f"{meta.width}x{meta.height}",
            },
            "events": [e.to_dict() for e in self._all_events],
            "summary": {
                "total_falls": len(self._all_events),
                "persons_tracked": len(unique_tracks),
                "processing_time_seconds": round(processing_time, 1),
                "interrupted": self._interrupted,
            },
        }

        # Save JSON
        if json_path:
            json_path = Path(json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Event log saved to {json_path}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Falls detected: {len(self._all_events)}")
        logger.info(f"Persons tracked: {len(unique_tracks)}")
        logger.info(f"Processing time: {processing_time:.1f}s")
        if output_path:
            logger.info(f"Output video: {output_path}")
        if json_path:
            logger.info(f"Event log: {json_path}")

        return result

    @staticmethod
    def _draw_label_scaled(frame, bbox, label, color, state, scale):
        """Draw a large, easy-to-read label above the bounding box."""
        import cv2

        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_thick = max(2, int(3 * scale))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thick)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_val = 0.7 * scale
        font_thick = max(1, int(2 * scale))
        padding = int(6 * scale)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale_val, font_thick)

        # Position label above bbox, or below if no room
        if y1 - th - baseline - padding * 2 >= 0:
            lbl_y2 = y1
            lbl_y1 = y1 - th - baseline - padding * 2
        else:
            lbl_y1 = y1
            lbl_y2 = y1 + th + baseline + padding * 2

        lbl_x1 = x1
        lbl_x2 = x1 + tw + padding * 2

        # Background
        cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)

        # Text — white for fall/fallen states, black for normal
        text_color = (255, 255, 255) if state != "normal" else (0, 0, 0)
        if y1 - th - padding * 2 >= 0:
            text_y = lbl_y2 - baseline - padding
        else:
            text_y = lbl_y1 + th + padding
        cv2.putText(
            frame, label, (lbl_x1 + padding, text_y),
            font, font_scale_val, text_color, font_thick, cv2.LINE_AA,
        )

    def _classify(self, sequence: np.ndarray) -> tuple[str, float]:
        """Classify a feature sequence using the LSTM."""
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        preds, confs = self.lstm.predict(x)
        class_id = preds[0].item()
        confidence = confs[0].item()
        class_name = FallDetector.CLASS_NAMES.get(class_id, "normal")
        return class_name, confidence

    def _handle_fall_event(
        self, track_id: int, frame_num: int, state: str, conf: float, fps: float
    ):
        """Handle a detected fall, with cooldown logic."""
        # Check cooldown
        last_alert = self._cooldowns.get(track_id, -self.cooldown_frames - 1)
        if frame_num - last_alert < self.cooldown_frames:
            # Update existing active event
            if track_id in self._active_events:
                self._active_events[track_id].update(frame_num, conf, state)
            return

        if track_id not in self._active_events:
            # New fall event
            event = FallEvent(track_id, frame_num, fps, conf)
            self._active_events[track_id] = event
            self._all_events.append(event)
            self._cooldowns[track_id] = frame_num
            logger.info(
                f"FALL DETECTED: Track {track_id} at frame {frame_num} "
                f"({format_timestamp(frame_num, fps)}) conf={conf:.2f}"
            )
        else:
            self._active_events[track_id].update(frame_num, conf, state)

    def _handle_interrupt(self, signum, frame):
        self._interrupted = True
