#!/usr/bin/env python3
"""Prepare training data for the fall detection LSTM.

Processes real video datasets through YOLO+BoT-SORT pipeline, combines with
synthetic data, and produces labeled .npy training splits.

Usage:
    # Full pipeline: process real datasets + synthetic augmentation
    python scripts/prepare_data.py --ur-fall data/raw/ur_fall --le2i data/raw/le2i --output data/splits

    # Generate synthetic data only (no GPU/videos needed)
    python scripts/prepare_data.py --synthetic-only --output data/splits
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_processor import (
    augment_data,
    generate_synthetic_data,
    process_dataset,
    save_splits,
    split_data,
)
from src.label_parser import (
    parse_generic_folder,
    parse_le2i_dataset,
    parse_ur_fall_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_real_datasets(args) -> tuple[np.ndarray, np.ndarray] | None:
    """Process real video datasets through YOLO+BoT-SORT pipeline."""
    from src.detector import PersonDetector
    from src.feature_extraction import FeatureExtractor

    all_annotations = []

    # Parse UR Fall Detection Dataset
    if args.ur_fall:
        ur_path = Path(args.ur_fall)
        if ur_path.exists():
            annotations = parse_ur_fall_dataset(ur_path)
            all_annotations.extend(annotations)
            logger.info(f"UR Fall: {len(annotations)} videos found")
        else:
            logger.warning(f"UR Fall directory not found: {ur_path}")

    # Parse Le2i Dataset
    if args.le2i:
        le2i_path = Path(args.le2i)
        if le2i_path.exists():
            annotations = parse_le2i_dataset(le2i_path)
            all_annotations.extend(annotations)
            logger.info(f"Le2i: {len(annotations)} videos found")
        else:
            logger.warning(f"Le2i directory not found: {le2i_path}")

    # Parse any generic dataset directories
    if args.generic:
        for gdir in args.generic:
            g_path = Path(gdir)
            if g_path.exists():
                annotations = parse_generic_folder(g_path)
                all_annotations.extend(annotations)
                logger.info(f"Generic ({g_path.name}): {len(annotations)} videos found")

    if not all_annotations:
        logger.warning("No real dataset videos found!")
        return None

    logger.info(f"\nTotal videos to process: {len(all_annotations)}")
    falls = sum(1 for a in all_annotations if a.is_fall)
    adl = len(all_annotations) - falls
    logger.info(f"  Falls: {falls}, ADL/Normal: {adl}")

    # Initialize detector and feature extractor
    logger.info("\nInitializing YOLO pose detector...")
    detector = PersonDetector(
        model_name=args.yolo_model,
        device=args.device,
        confidence_threshold=0.3,
        tracker_config="configs/botsort.yaml",
    )
    feature_extractor = FeatureExtractor()

    # Process all videos
    logger.info("Processing videos through YOLO+BoT-SORT pipeline...")
    X_real, y_real = process_dataset(
        annotations=all_annotations,
        detector=detector,
        feature_extractor=feature_extractor,
        window_size=args.window_size,
        stride=args.stride,
    )

    if len(X_real) == 0:
        logger.warning("No feature sequences extracted from real videos!")
        return None

    logger.info(f"\nReal data: {X_real.shape[0]} sequences extracted")
    return X_real, y_real


def main():
    parser = argparse.ArgumentParser(description="Prepare fall detection training data")

    # Dataset paths
    parser.add_argument("--ur-fall", type=str, default=None, help="Path to UR Fall Detection dataset")
    parser.add_argument("--le2i", type=str, default=None, help="Path to Le2i Fall dataset")
    parser.add_argument("--generic", type=str, nargs="*", default=None, help="Paths to generic fall/normal dirs")

    # Output
    parser.add_argument("--output", type=str, default="data/splits", help="Output directory for .npy files")

    # Options
    parser.add_argument("--synthetic-only", action="store_true", help="Generate only synthetic data")
    parser.add_argument("--num-synthetic", type=int, default=3000, help="Number of synthetic samples")
    parser.add_argument("--window-size", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--stride", type=int, default=15, help="Stride between windows")
    parser.add_argument("--augment-factor", type=int, default=2, help="Augmentation multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model options
    parser.add_argument("--yolo-model", type=str, default="yolo11n-pose.pt", help="YOLO pose model")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Fall Detection — Data Preparation Pipeline")
    logger.info("=" * 60)

    X_parts = []
    y_parts = []

    # --- Step 1: Process real datasets ---
    if not args.synthetic_only:
        result = process_real_datasets(args)
        if result is not None:
            X_real, y_real = result
            X_parts.append(X_real)
            y_parts.append(y_real)
            logger.info(f"Real data: {len(X_real)} samples")
    else:
        logger.info("Skipping real datasets (--synthetic-only)")

    # --- Step 2: Generate synthetic data ---
    logger.info(f"\nGenerating {args.num_synthetic} synthetic samples...")
    X_syn, y_syn = generate_synthetic_data(
        num_samples=args.num_synthetic,
        window_size=args.window_size,
        seed=args.seed,
    )
    X_parts.append(X_syn)
    y_parts.append(y_syn)
    logger.info(f"Synthetic data: {len(X_syn)} samples")

    # --- Step 3: Combine ---
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(f"\nCombined data: {len(X)} total samples")

    # --- Step 4: Augment ---
    logger.info(f"Augmenting data (factor={args.augment_factor})...")
    X, y = augment_data(X, y, augment_factor=args.augment_factor, seed=args.seed)
    logger.info(f"After augmentation: {len(X)} samples")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: "normal", 1: "falling", 2: "fallen"}
    logger.info("\nClass distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100
        logger.info(f"  Class {label} ({label_names.get(label, '?')}): {count} ({pct:.1f}%)")

    # --- Step 5: Split ---
    logger.info("\nSplitting into train/val/test (80/10/10, stratified)...")
    splits = split_data(X, y, train_ratio=0.8, val_ratio=0.1, seed=args.seed)

    for name, (X_s, y_s) in splits.items():
        unique_s, counts_s = np.unique(y_s, return_counts=True)
        logger.info(f"  {name}: X={X_s.shape}, classes={dict(zip(unique_s.tolist(), counts_s.tolist()))}")

    # --- Step 6: Validate ---
    logger.info("\nValidating data quality...")
    for name, (X_s, y_s) in splits.items():
        assert not np.any(np.isnan(X_s)), f"NaN found in {name} features"
        assert not np.any(np.isinf(X_s)), f"Inf found in {name} features"
        assert len(X_s) == len(y_s), f"Shape mismatch in {name}"
        assert X_s.shape[1] == args.window_size, f"Wrong window size in {name}"
        assert X_s.shape[2] == 5, f"Wrong feature count in {name}"
    logger.info("  All validation checks passed!")

    # --- Step 7: Save ---
    save_splits(splits, args.output)
    logger.info(f"\nData saved to {args.output}/")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    total_train = len(splits["train"][0])
    total_val = len(splits["val"][0])
    total_test = len(splits["test"][0])
    logger.info(f"Train: {total_train} | Val: {total_val} | Test: {total_test}")
    logger.info(f"Total: {total_train + total_val + total_test}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
