#!/usr/bin/env python3
"""Human-friendly feature visualization for fall detection.

Shows features in plain language with visual indicators that anyone can understand.
No normalized numbers — uses real-world descriptions and color-coded bars.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.fall_detector import FallDetector
from src.detector import PersonDetector
from src.feature_extraction import FeatureExtractor
from src.track_manager import TrackManager
from src.video_io import VideoReader

# Colors (BGR)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
DARK_GRAY = (80, 80, 80)
YELLOW = (0, 230, 230)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)


def get_angle_description(angle_normalized):
    """Convert normalized angle to human-readable description."""
    deg = angle_normalized * 180
    if deg < 20:
        return f"{deg:.0f} deg", "Upright", GREEN, deg / 90
    elif deg < 45:
        return f"{deg:.0f} deg", "Leaning", YELLOW, deg / 90
    elif deg < 70:
        return f"{deg:.0f} deg", "Falling", ORANGE, deg / 90
    else:
        return f"{deg:.0f} deg", "Lying Down", RED, min(deg / 90, 1.0)


def get_cog_description(cog_height):
    """Convert CoG height to human-readable description."""
    if cog_height < 0.45:
        return "High (Standing)", GREEN, cog_height
    elif cog_height < 0.65:
        return "Middle", YELLOW, cog_height
    elif cog_height < 0.80:
        return "Low (Crouching)", ORANGE, cog_height
    else:
        return "Very Low (Ground)", RED, cog_height


def get_velocity_description(velocity):
    """Convert velocity to human-readable description."""
    v = abs(velocity)
    if v < 0.005:
        return "Stable", GREEN, v / 0.05
    elif v < 0.02:
        return "Moving Slowly", YELLOW, v / 0.05
    elif v < 0.05:
        return "Moving Fast", ORANGE, v / 0.05
    else:
        return "Rapid Drop!", RED, min(v / 0.05, 1.0)


def get_ratio_description(ratio):
    """Convert bbox aspect ratio to human-readable description."""
    if ratio < 0.7:
        return "Tall & Narrow", GREEN, ratio / 3.0
    elif ratio < 1.2:
        return "Square", YELLOW, ratio / 3.0
    elif ratio < 2.0:
        return "Wide", ORANGE, ratio / 3.0
    else:
        return "Very Wide (Flat)", RED, min(ratio / 3.0, 1.0)


def get_confidence_description(conf):
    """Convert keypoint confidence to human-readable description."""
    if conf > 0.85:
        return "Clear Detection", GREEN, conf
    elif conf > 0.65:
        return "Moderate", YELLOW, conf
    elif conf > 0.4:
        return "Partial View", ORANGE, conf
    else:
        return "Weak Detection", RED, conf


def draw_feature_panel_v2(frame, features, state, conf, track_id, scale=1.0):
    """Draw a human-friendly feature panel with plain language descriptions."""
    h, w = frame.shape[:2]

    panel_w = int(420 * scale)
    panel_h = int(430 * scale)
    panel_x = w - panel_w - int(15 * scale)
    panel_y = int(15 * scale)

    # Dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Border
    if state == "normal":
        border_color = GREEN
        state_text = "NORMAL"
    elif state == "falling":
        border_color = RED
        state_text = "FALLING"
    else:
        border_color = ORANGE
        state_text = "FALLEN"

    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  border_color, max(2, int(3 * scale)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = panel_x + int(15 * scale)
    y = panel_y + int(32 * scale)

    # Title
    fs_title = 0.65 * scale
    ft_title = max(2, int(2 * scale))
    cv2.putText(frame, "FEATURE ANALYSIS", (x, y),
                font, fs_title, WHITE, ft_title, cv2.LINE_AA)
    y += int(26 * scale)

    # State badge
    badge_w = int(len(state_text) * 16 * scale)
    cv2.rectangle(frame, (x, y - int(18 * scale)),
                  (x + badge_w, y + int(6 * scale)), border_color, -1)
    cv2.putText(frame, state_text, (x + int(5 * scale), y),
                font, 0.55 * scale, (0, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)

    conf_text = f"  Confidence: {conf:.0%}"
    cv2.putText(frame, conf_text, (x + badge_w + int(5 * scale), y),
                font, 0.45 * scale, GRAY, 1, cv2.LINE_AA)
    y += int(18 * scale)

    # Divider
    cv2.line(frame, (x, y), (panel_x + panel_w - int(15 * scale), y), DARK_GRAY, 1)
    y += int(18 * scale)

    # Feature rows
    fs = 0.48 * scale
    ft = max(1, int(1.5 * scale))
    bar_w = int(150 * scale)
    bar_h = int(14 * scale)
    row_h = int(72 * scale)
    val_x = panel_x + panel_w - bar_w - int(20 * scale)

    # 1. Body Angle
    deg_text, angle_desc, angle_color, angle_fill = get_angle_description(features.body_angle)
    _draw_feature_row(frame, x, y, "Body Posture", deg_text, angle_desc,
                      angle_color, angle_fill, bar_w, bar_h, val_x, scale, font, fs, ft)
    y += row_h

    # 2. CoG Height
    cog_desc, cog_color, cog_fill = get_cog_description(features.cog_height)
    _draw_feature_row(frame, x, y, "Body Height", f"{features.cog_height * 100:.0f}%",
                      cog_desc, cog_color, cog_fill, bar_w, bar_h, val_x, scale, font, fs, ft)
    y += row_h

    # 3. Velocity
    vel_desc, vel_color, vel_fill = get_velocity_description(features.vertical_velocity)
    direction = "Down" if features.vertical_velocity > 0.003 else "Up" if features.vertical_velocity < -0.003 else ""
    vel_label = f"{direction} " if direction else ""
    _draw_feature_row(frame, x, y, "Movement Speed", vel_label + vel_desc, vel_desc,
                      vel_color, vel_fill, bar_w, bar_h, val_x, scale, font, fs, ft)
    y += row_h

    # 4. Bbox Ratio
    ratio_desc, ratio_color, ratio_fill = get_ratio_description(features.bbox_aspect_ratio)
    _draw_feature_row(frame, x, y, "Body Shape", f"{features.bbox_aspect_ratio:.1f}x",
                      ratio_desc, ratio_color, ratio_fill, bar_w, bar_h, val_x, scale, font, fs, ft)
    y += row_h

    # 5. Keypoint Confidence
    kp_desc, kp_color, kp_fill = get_confidence_description(features.keypoint_confidence)
    _draw_feature_row(frame, x, y, "Detection Quality", f"{features.keypoint_confidence * 100:.0f}%",
                      kp_desc, kp_color, kp_fill, bar_w, bar_h, val_x, scale, font, fs, ft)

    return frame


def _draw_feature_row(frame, x, y, title, value_text, description,
                      color, fill_fraction, bar_w, bar_h, val_x, scale, font, fs, ft):
    """Draw a single feature row with title, description, value, and color bar."""
    # Feature title
    cv2.putText(frame, title, (x, y), font, fs * 1.05, WHITE, ft, cv2.LINE_AA)

    # Description text (colored)
    y_desc = y + int(20 * scale)
    cv2.putText(frame, description, (x, y_desc), font, fs * 0.9, color, ft, cv2.LINE_AA)

    # Value text
    cv2.putText(frame, value_text, (val_x, y), font, fs, GRAY, 1, cv2.LINE_AA)

    # Color bar
    bar_y = y_desc + int(6 * scale)
    fill_w = int(fill_fraction * bar_w)
    cv2.rectangle(frame, (val_x, bar_y), (val_x + bar_w, bar_y + bar_h), DARK_GRAY, -1)
    if fill_w > 0:
        cv2.rectangle(frame, (val_x, bar_y), (val_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (val_x, bar_y), (val_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)


def draw_skeleton_annotated(frame, keypoints, bbox, features, scale=1.0):
    """Draw skeleton with body angle arc and CoG marker."""
    NOSE, LEFT_HIP, RIGHT_HIP = 0, 11, 12
    conf_thresh = 0.3
    kps = keypoints

    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    thick = max(2, int(3 * scale))
    for i, j in connections:
        if i < len(kps) and j < len(kps):
            if kps[i][2] >= conf_thresh and kps[j][2] >= conf_thresh:
                cv2.line(frame, (int(kps[i][0]), int(kps[i][1])),
                         (int(kps[j][0]), int(kps[j][1])), CYAN, thick)

    for kp in kps:
        if kp[2] >= conf_thresh:
            cv2.circle(frame, (int(kp[0]), int(kp[1])),
                       max(4, int(5 * scale)), CYAN, -1)

    # Body angle visualization
    nose = kps[NOSE]
    lhip, rhip = kps[LEFT_HIP], kps[RIGHT_HIP]

    if nose[2] >= conf_thresh and (lhip[2] >= conf_thresh or rhip[2] >= conf_thresh):
        if lhip[2] >= conf_thresh and rhip[2] >= conf_thresh:
            mid = (int((lhip[0] + rhip[0]) / 2), int((lhip[1] + rhip[1]) / 2))
        elif lhip[2] >= conf_thresh:
            mid = (int(lhip[0]), int(lhip[1]))
        else:
            mid = (int(rhip[0]), int(rhip[1]))

        nose_pt = (int(nose[0]), int(nose[1]))

        # Body line (hip→nose)
        cv2.line(frame, mid, nose_pt, MAGENTA, max(2, int(3 * scale)))

        # Vertical reference
        vert_len = int(80 * scale)
        cv2.line(frame, mid, (mid[0], mid[1] - vert_len), (200, 200, 200), 1)

        # Angle arc
        deg = features.body_angle * 180
        if deg > 5:
            radius = int(35 * scale)
            cv2.ellipse(frame, mid, (radius, radius), 0, -90, -90 + deg,
                        MAGENTA, max(1, int(2 * scale)))
            # Degree label
            label_x = mid[0] + int(radius * 1.3)
            label_y = mid[1] - int(radius * 0.3)
            cv2.putText(frame, f"{deg:.0f} deg", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, MAGENTA,
                        max(1, int(2 * scale)), cv2.LINE_AA)

        # CoG marker
        r = max(8, int(10 * scale))
        cv2.circle(frame, mid, r, GREEN, max(2, int(3 * scale)))
        cv2.line(frame, (mid[0] - r, mid[1]), (mid[0] + r, mid[1]), GREEN, 2)
        cv2.line(frame, (mid[0], mid[1] - r), (mid[0], mid[1] + r), GREEN, 2)
        cv2.putText(frame, "CoG", (mid[0] + r + 5, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, GREEN,
                    max(1, int(2 * scale)), cv2.LINE_AA)

    # Bbox with ratio
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, max(1, int(2 * scale)))
    cv2.putText(frame, f"W/H = {features.bbox_aspect_ratio:.2f}",
                (x1, y2 + int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * scale, YELLOW, max(1, int(2 * scale)), cv2.LINE_AA)

    return frame


def process_video(video_path, normal_frame, fallen_frame, output_prefix):
    """Process video and extract annotated frames."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = PersonDetector(
        device=device, confidence_threshold=0.3, tracker_config="configs/botsort.yaml"
    )
    fe = FeatureExtractor()
    tm = TrackManager(window_size=30, stale_threshold=90)

    lstm = FallDetector()
    ckpt = torch.load("models/checkpoints/best.pth", map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.to(device)
    lstm.eval()

    fall_detected_frame = None

    with VideoReader(video_path) as reader:
        h = reader.metadata.height
        scale = max(h / 720, 1.0)

        for frame_num, frame in reader:
            result = detector.detect_frame(frame, frame_num)

            for person in result.persons:
                if person.track_id < 0:
                    continue

                prev_cog = tm.get_prev_cog_height(person.track_id)
                features = fe.extract(person.keypoints, person.bbox, h, prev_cog)
                tm.update(person.track_id, features, frame_num)

                seq = tm.get_sequence(person.track_id)
                state, conf_val = "normal", 0.0
                if seq is not None:
                    x = torch.FloatTensor(seq).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(lstm(x), dim=1).cpu().numpy()[0]
                    pred = probs.argmax()
                    conf_val = float(probs[pred])
                    state = {0: "normal", 1: "falling", 2: "fallen"}[pred]

                if state in ("falling", "fallen") and fall_detected_frame is None:
                    fall_detected_frame = frame_num

                if frame_num in (normal_frame, fallen_frame):
                    annotated = frame.copy()
                    draw_skeleton_annotated(annotated, person.keypoints,
                                           person.bbox, features, scale)
                    draw_feature_panel_v2(annotated, features, state,
                                         conf_val, person.track_id, scale)

                    tag = "NORMAL" if frame_num == normal_frame else "FALLEN"
                    path = f"{output_prefix}_{tag.lower()}.png"
                    cv2.imwrite(path, annotated)
                    print(f"  Saved {tag}: {path}")


if __name__ == "__main__":
    out = "output/test_results"

    print("=== Elderly Falling ===")
    process_video(
        "tests/test_videos/elderly_falling.mp4",
        normal_frame=50, fallen_frame=165,
        output_prefix=f"{out}/features_v2_elderly",
    )

    print("\n=== Skating ===")
    process_video(
        "tests/test_videos/skating.mp4",
        normal_frame=40, fallen_frame=155,
        output_prefix=f"{out}/features_v2_skating",
    )
