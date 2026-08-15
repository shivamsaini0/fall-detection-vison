#!/usr/bin/env python3
"""Visualize the 5 extracted features on video frames.

Creates annotated screenshots showing feature values for NORMAL vs FALLEN states,
making the feature extraction process visually clear for portfolio/presentation.
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


def draw_feature_panel(frame, features, state, conf, track_id, scale=1.0):
    """Draw a professional feature info panel on the right side of the frame."""
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = int(380 * scale)
    panel_h = int(380 * scale)
    panel_x = w - panel_w - int(15 * scale)
    panel_y = int(15 * scale)

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border color based on state
    border_color = (0, 255, 0) if state == "normal" else (0, 0, 255) if state == "falling" else (0, 165, 255)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        border_color,
        max(2, int(3 * scale)),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55 * scale  # font scale
    ft = max(1, int(2 * scale))  # font thickness
    fs_title = 0.7 * scale
    ft_title = max(2, int(2 * scale))
    line_h = int(42 * scale)
    x = panel_x + int(15 * scale)
    y = panel_y + int(35 * scale)

    # Title
    state_upper = state.upper()
    cv2.putText(frame, f"FEATURE EXTRACTION", (x, y), font, fs_title, (255, 255, 255), ft_title, cv2.LINE_AA)
    y += int(28 * scale)
    cv2.putText(frame, f"ID:{track_id} | State: {state_upper} ({conf:.0%})", (x, y), font, fs * 0.9, border_color, ft, cv2.LINE_AA)
    y += int(15 * scale)

    # Divider line
    cv2.line(frame, (x, y), (panel_x + panel_w - int(15 * scale), y), (100, 100, 100), 1)
    y += int(22 * scale)

    # Feature values with visual bars
    bar_max_w = int(160 * scale)
    label_x = x
    value_x = x + int(185 * scale)
    bar_x = value_x

    feature_data = [
        ("Body Angle", features.body_angle, 1.0, "deg",
         f"{features.body_angle * 180:.0f}" + chr(176),
         "0=standing, 90=lying"),
        ("CoG Height", features.cog_height, 1.0, "",
         f"{features.cog_height:.2f}",
         "0=top, 1=ground"),
        ("Velocity", abs(features.vertical_velocity), 0.1, "",
         f"{features.vertical_velocity:+.3f}",
         "+down, -up"),
        ("Bbox Ratio", features.bbox_aspect_ratio, 3.0, "",
         f"{features.bbox_aspect_ratio:.2f}",
         "tall<1, wide>1"),
        ("KP Confidence", features.keypoint_confidence, 1.0, "",
         f"{features.keypoint_confidence:.2f}",
         "0=low, 1=high"),
    ]

    for name, value, max_val, unit, display_val, hint in feature_data:
        # Feature name
        cv2.putText(frame, name, (label_x, y), font, fs * 0.85, (200, 200, 200), ft, cv2.LINE_AA)

        # Value
        val_color = (255, 255, 255)
        cv2.putText(frame, display_val, (value_x, y), font, fs, val_color, ft, cv2.LINE_AA)

        # Bar
        bar_y = y + int(6 * scale)
        bar_h = int(10 * scale)
        bar_fill = min(value / max_val, 1.0) if max_val > 0 else 0
        fill_w = int(bar_fill * bar_max_w)

        # Bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 60), -1)
        # Bar fill
        if fill_w > 0:
            bar_color = border_color
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Hint text
        hint_y = bar_y + bar_h + int(14 * scale)
        cv2.putText(frame, hint, (bar_x, hint_y), font, fs * 0.6, (120, 120, 120), 1, cv2.LINE_AA)

        y += line_h

    return frame


def draw_skeleton_with_annotations(frame, keypoints, bbox, features, scale=1.0):
    """Draw skeleton with body angle line and CoG marker."""
    # COCO indices
    NOSE = 0
    LEFT_HIP = 11
    RIGHT_HIP = 12

    conf_thresh = 0.3
    kps = keypoints

    # Draw skeleton connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    skel_color = (0, 255, 255)  # Cyan for visibility
    thick = max(2, int(3 * scale))

    for i, j in connections:
        if i < len(kps) and j < len(kps):
            if kps[i][2] >= conf_thresh and kps[j][2] >= conf_thresh:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, skel_color, thick)

    for kp in kps:
        if kp[2] >= conf_thresh:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), max(4, int(5 * scale)), skel_color, -1)

    # Draw body angle line (hip midpoint → nose) with angle arc
    nose = kps[NOSE]
    lhip = kps[LEFT_HIP]
    rhip = kps[RIGHT_HIP]

    has_nose = nose[2] >= conf_thresh
    has_lhip = lhip[2] >= conf_thresh
    has_rhip = rhip[2] >= conf_thresh

    if has_nose and (has_lhip or has_rhip):
        if has_lhip and has_rhip:
            mid_x = (lhip[0] + rhip[0]) / 2
            mid_y = (lhip[1] + rhip[1]) / 2
        elif has_lhip:
            mid_x, mid_y = lhip[0], lhip[1]
        else:
            mid_x, mid_y = rhip[0], rhip[1]

        mid = (int(mid_x), int(mid_y))
        nose_pt = (int(nose[0]), int(nose[1]))

        # Draw body angle line (hip→nose) in magenta
        cv2.line(frame, mid, nose_pt, (255, 0, 255), max(2, int(3 * scale)))

        # Draw vertical reference line from hip midpoint (white dashed)
        vert_top = (mid[0], mid[1] - 100)
        cv2.line(frame, mid, vert_top, (255, 255, 255), 1)

        # Draw angle arc
        angle_deg = features.body_angle * 180
        if angle_deg > 5:
            radius = int(40 * scale)
            # Arc from vertical (90 deg in OpenCV) sweeping by body angle
            start_angle = -90
            end_angle = start_angle + angle_deg
            cv2.ellipse(
                frame, mid, (radius, radius),
                0, start_angle, end_angle,
                (255, 0, 255), max(1, int(2 * scale)),
            )
            # Label the angle
            arc_label_x = mid[0] + int(radius * 1.2)
            arc_label_y = mid[1] - int(radius * 0.5)
            cv2.putText(
                frame, f"{angle_deg:.0f}" + chr(176),
                (arc_label_x, arc_label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 0, 255),
                max(1, int(2 * scale)), cv2.LINE_AA,
            )

    # Draw CoG marker (center of gravity = hip midpoint)
    if has_lhip or has_rhip:
        if has_lhip and has_rhip:
            cog_x = int((lhip[0] + rhip[0]) / 2)
            cog_y = int((lhip[1] + rhip[1]) / 2)
        elif has_lhip:
            cog_x, cog_y = int(lhip[0]), int(lhip[1])
        else:
            cog_x, cog_y = int(rhip[0]), int(rhip[1])

        r = max(8, int(10 * scale))
        cv2.circle(frame, (cog_x, cog_y), r, (0, 255, 0), max(2, int(3 * scale)))
        cv2.line(frame, (cog_x - r, cog_y), (cog_x + r, cog_y), (0, 255, 0), 2)
        cv2.line(frame, (cog_x, cog_y - r), (cog_x, cog_y + r), (0, 255, 0), 2)
        cv2.putText(
            frame, "CoG",
            (cog_x + r + 5, cog_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 255, 0),
            max(1, int(2 * scale)), cv2.LINE_AA,
        )

    # Draw bbox with aspect ratio annotation
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), max(1, int(2 * scale)))
    ratio_label = f"W/H={features.bbox_aspect_ratio:.2f}"
    cv2.putText(
        frame, ratio_label,
        (x1, y2 + int(20 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 0),
        max(1, int(2 * scale)), cv2.LINE_AA,
    )

    return frame


def process_video_extract_frames(video_path, normal_frame, fallen_frame, output_prefix):
    """Process a video and extract annotated frames at specific positions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = PersonDetector(device=device, confidence_threshold=0.3, tracker_config="configs/botsort.yaml")
    fe = FeatureExtractor()
    tm = TrackManager(window_size=30, stale_threshold=90)

    lstm = FallDetector()
    ckpt = torch.load("models/checkpoints/best.pth", map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.to(device)
    lstm.eval()

    saved_frames = {}

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
                state = "normal"
                conf = 0.0
                if seq is not None:
                    x = torch.FloatTensor(seq).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(lstm(x), dim=1).cpu().numpy()[0]
                    pred = probs.argmax()
                    conf = probs[pred]
                    state = {0: "normal", 1: "falling", 2: "fallen"}[pred]

                if frame_num in (normal_frame, fallen_frame):
                    annotated = frame.copy()
                    draw_skeleton_with_annotations(
                        annotated, person.keypoints, person.bbox, features, scale
                    )
                    draw_feature_panel(annotated, features, state, conf, person.track_id, scale)

                    tag = "NORMAL" if frame_num == normal_frame else "FALLEN"
                    path = f"{output_prefix}_{tag.lower()}_f{frame_num}.png"
                    cv2.imwrite(path, annotated)
                    saved_frames[tag] = path
                    print(f"Saved {tag}: {path}")
                    print(f"  Features: angle={features.body_angle*180:.1f}° "
                          f"cog={features.cog_height:.3f} "
                          f"vel={features.vertical_velocity:+.4f} "
                          f"ratio={features.bbox_aspect_ratio:.3f} "
                          f"conf={features.keypoint_confidence:.3f}")

    return saved_frames


if __name__ == "__main__":
    output_dir = "output/test_results"

    print("=== Elderly Falling ===")
    process_video_extract_frames(
        "tests/test_videos/elderly_falling.mp4",
        normal_frame=50,
        fallen_frame=165,
        output_prefix=f"{output_dir}/features_elderly",
    )

    print("\n=== Skating ===")
    process_video_extract_frames(
        "tests/test_videos/skating.mp4",
        normal_frame=40,
        fallen_frame=155,
        output_prefix=f"{output_dir}/features_skating",
    )
