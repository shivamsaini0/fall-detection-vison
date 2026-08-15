#!/usr/bin/env python3
"""Batch test fall detection on all test videos and generate a summary report.

Usage:
    python scripts/batch_test.py --input tests/test_videos/ --output output/test_results/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import FallDetectionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description="Batch test fall detection")
    parser.add_argument("--input", "-i", default="tests/test_videos/",
                        help="Directory with test videos")
    parser.add_argument("--output", "-o", default="output/test_results/",
                        help="Output directory for results")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all test videos
    videos = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        logger.error(f"No video files found in {input_dir}")
        return

    logger.info(f"Found {len(videos)} test videos in {input_dir}")
    logger.info("=" * 70)

    # Initialize pipeline once
    pipeline = FallDetectionPipeline(
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Process each video
    all_results = []
    total_start = time.time()

    for i, video_path in enumerate(videos, 1):
        name = video_path.stem
        logger.info(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")

        out_video = output_dir / f"{name}_result.mp4"
        out_json = output_dir / f"{name}_events.json"

        try:
            result = pipeline.process_video(
                input_path=video_path,
                output_path=out_video,
                json_path=out_json,
            )
            result["video_name"] = video_path.name
            result["status"] = "success"
            all_results.append(result)

        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            all_results.append({
                "video_name": video_path.name,
                "status": "error",
                "error": str(e),
            })

    total_time = time.time() - total_start

    # Generate summary report
    logger.info("\n" + "=" * 70)
    logger.info("BATCH TEST SUMMARY")
    logger.info("=" * 70)

    summary = {
        "total_videos": len(videos),
        "successful": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "error"),
        "total_processing_time": round(total_time, 1),
        "confidence_threshold": args.confidence,
        "results": [],
    }

    for r in all_results:
        if r["status"] == "success":
            entry = {
                "video": r["video_name"],
                "falls_detected": r["summary"]["total_falls"],
                "persons_tracked": r["summary"]["persons_tracked"],
                "processing_time": r["summary"]["processing_time_seconds"],
                "events": r["events"],
            }
            summary["results"].append(entry)

            falls = r["summary"]["total_falls"]
            persons = r["summary"]["persons_tracked"]
            ptime = r["summary"]["processing_time_seconds"]
            logger.info(
                f"  {r['video_name']:<35} | "
                f"Falls: {falls} | Persons: {persons} | Time: {ptime:.1f}s"
            )
        else:
            summary["results"].append({
                "video": r["video_name"],
                "status": "error",
                "error": r.get("error", "unknown"),
            })
            logger.info(f"  {r['video_name']:<35} | ERROR: {r.get('error', 'unknown')}")

    total_falls = sum(
        r.get("summary", {}).get("total_falls", 0)
        for r in all_results if r["status"] == "success"
    )
    logger.info(f"\nTotal falls detected across all videos: {total_falls}")
    logger.info(f"Total processing time: {total_time:.1f}s")

    # Save summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"All results in {output_dir}/")


if __name__ == "__main__":
    main()
