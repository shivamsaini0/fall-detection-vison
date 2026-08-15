#!/usr/bin/env python3
"""Fall Detection CLI — process a video and detect falls.

Usage:
    python detect_falls.py --input video.mp4 --output result.mp4 --json-log events.json
"""

import argparse
import logging
import sys

from src.pipeline import FallDetectionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Detect falls in video using skeleton tracking + LSTM"
    )
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", default=None, help="Output annotated video path")
    parser.add_argument("--json-log", "-j", default=None, help="JSON event log output path")
    parser.add_argument(
        "--model", default="models/checkpoints/best.pth", help="LSTM model checkpoint"
    )
    parser.add_argument("--yolo-model", default="yolo11n-pose.pt", help="YOLO pose model")
    parser.add_argument("--confidence", type=float, default=0.40, help="Fall confidence threshold")
    parser.add_argument("--cooldown", type=int, default=90, help="Cooldown frames between alerts")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto")

    args = parser.parse_args()

    pipeline = FallDetectionPipeline(
        yolo_model=args.yolo_model,
        lstm_checkpoint=args.model,
        device=args.device,
        confidence_threshold=args.confidence,
        cooldown_frames=args.cooldown,
    )

    result = pipeline.process_video(
        input_path=args.input,
        output_path=args.output,
        json_path=args.json_log,
    )

    return 0 if not result["summary"]["interrupted"] else 1


if __name__ == "__main__":
    sys.exit(main())
