"""Batch process fall detection videos into labeled training data (.npy)."""

import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

from src.detector import PersonDetector
from src.feature_extraction import NUM_FEATURES, FeatureExtractor
from src.label_parser import VideoAnnotation
from src.track_manager import TrackManager
from src.video_io import ImageSequenceReader, VideoReader

logger = logging.getLogger(__name__)


def process_video_to_features(
    video_path: str | Path,
    detector: PersonDetector,
    feature_extractor: FeatureExtractor,
    window_size: int = 30,
    stride: int = 15,
) -> list[np.ndarray]:
    """Process a single video into feature sequences.

    Args:
        video_path: Path to the video file.
        detector: PersonDetector instance.
        feature_extractor: FeatureExtractor instance.
        window_size: Number of frames per sequence.
        stride: Step between consecutive windows.

    Returns:
        List of numpy arrays, each of shape (window_size, NUM_FEATURES).
    """
    track_manager = TrackManager(window_size=window_size * 2)
    all_track_features: dict[int, list[np.ndarray]] = {}

    try:
        with VideoReader(video_path) as reader:
            frame_height = reader.metadata.height

            for frame_num, frame in reader:
                # Detect and track persons
                frame_result = detector.detect_frame(frame, frame_num)

                for person in frame_result.persons:
                    if person.track_id < 0:
                        continue

                    prev_cog = track_manager.get_prev_cog_height(person.track_id)
                    features = feature_extractor.extract(
                        person.keypoints, person.bbox, frame_height, prev_cog
                    )
                    track_manager.update(person.track_id, features, frame_num)

                    # Store raw feature arrays per track
                    if person.track_id not in all_track_features:
                        all_track_features[person.track_id] = []
                    all_track_features[person.track_id].append(features.to_array())

                # Cleanup stale tracks periodically
                if frame_num % 100 == 0:
                    track_manager.cleanup(frame_num)

    except (IOError, ValueError) as e:
        logger.warning(f"Failed to process video {video_path}: {e}")
        return []

    # Slice each track's features into windows
    sequences = []
    for track_id, feature_list in all_track_features.items():
        if len(feature_list) < window_size:
            continue

        track_array = np.array(feature_list, dtype=np.float32)  # (T, 5)
        for start in range(0, len(track_array) - window_size + 1, stride):
            window = track_array[start : start + window_size]
            sequences.append(window)

    detector.reset_tracker()
    return sequences


def process_image_sequence_to_features(
    sequence_dir: str | Path,
    detector: PersonDetector,
    feature_extractor: FeatureExtractor,
    window_size: int = 30,
    stride: int = 15,
    fps: float = 30.0,
) -> list[np.ndarray]:
    """Process a folder of numbered images (PNG/JPG) into feature sequences.

    Used for datasets like UR Fall that store frames as individual images.

    Args:
        sequence_dir: Path to directory containing numbered image files.
        detector: PersonDetector instance.
        feature_extractor: FeatureExtractor instance.
        window_size: Number of frames per sequence.
        stride: Step between consecutive windows.
        fps: Assumed FPS of the image sequence.

    Returns:
        List of numpy arrays, each of shape (window_size, NUM_FEATURES).
    """
    track_manager = TrackManager(window_size=window_size * 2)
    all_track_features: dict[int, list[np.ndarray]] = {}

    try:
        with ImageSequenceReader(sequence_dir, fps=fps) as reader:
            frame_height = reader.metadata.height

            for frame_num, frame in reader:
                frame_result = detector.detect_frame(frame, frame_num)

                for person in frame_result.persons:
                    if person.track_id < 0:
                        continue

                    prev_cog = track_manager.get_prev_cog_height(person.track_id)
                    features = feature_extractor.extract(
                        person.keypoints, person.bbox, frame_height, prev_cog
                    )
                    track_manager.update(person.track_id, features, frame_num)

                    if person.track_id not in all_track_features:
                        all_track_features[person.track_id] = []
                    all_track_features[person.track_id].append(features.to_array())

                if frame_num % 100 == 0:
                    track_manager.cleanup(frame_num)

    except (IOError, ValueError) as e:
        logger.warning(f"Failed to process image sequence {sequence_dir}: {e}")
        return []

    sequences = []
    for track_id, feature_list in all_track_features.items():
        if len(feature_list) < window_size:
            continue
        track_array = np.array(feature_list, dtype=np.float32)
        for start in range(0, len(track_array) - window_size + 1, stride):
            window = track_array[start : start + window_size]
            sequences.append(window)

    detector.reset_tracker()
    return sequences


def assign_labels(
    num_windows: int,
    annotation: VideoAnnotation,
    window_size: int,
    stride: int,
    total_frames: int,
) -> np.ndarray:
    """Assign labels to sliding windows based on annotation.

    Labels:
        0 = normal
        1 = falling (transition period)
        2 = fallen (on ground after fall)

    Args:
        num_windows: Number of windows to label.
        annotation: Video annotation with fall frame ranges.
        window_size: Window size in frames.
        stride: Stride between windows.
        total_frames: Total frames in the video.

    Returns:
        Array of shape (num_windows,) with integer labels.
    """
    labels = np.zeros(num_windows, dtype=np.int64)

    if not annotation.is_fall:
        return labels  # All normal

    fall_start = annotation.fall_start_frame
    fall_end = annotation.fall_end_frame

    if fall_start is None:
        # No frame-level annotation; label all windows as fall
        labels[:] = 1
        return labels

    if fall_end is None:
        fall_end = total_frames

    for i in range(num_windows):
        window_start = i * stride
        window_end = window_start + window_size

        # Count how many frames in this window overlap with fall
        overlap_start = max(window_start, fall_start)
        overlap_end = min(window_end, fall_end)
        overlap = max(0, overlap_end - overlap_start)
        overlap_fraction = overlap / window_size

        if overlap_fraction > 0.5:
            # Majority of window is during fall
            # Distinguish falling (early) vs fallen (late)
            window_center = (window_start + window_end) / 2
            fall_duration = fall_end - fall_start
            if fall_duration > 0 and (window_center - fall_start) / fall_duration < 0.4:
                labels[i] = 1  # falling
            else:
                labels[i] = 2  # fallen/on-ground

    return labels


def process_dataset(
    annotations: list[VideoAnnotation],
    detector: PersonDetector,
    feature_extractor: FeatureExtractor,
    window_size: int = 30,
    stride: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Process an entire dataset of annotated videos into labeled training data.

    Runs YOLO-pose + BoT-SORT on every video, extracts features per person per frame,
    slices into sliding windows, and assigns labels based on annotations.

    Args:
        annotations: List of VideoAnnotation objects with video paths and labels.
        detector: PersonDetector instance (YOLO + BoT-SORT).
        feature_extractor: FeatureExtractor instance.
        window_size: Number of frames per sequence.
        stride: Step between consecutive windows.

    Returns:
        Tuple of (X, y) where X is (N, window_size, NUM_FEATURES) and y is (N,).
    """
    all_X = []
    all_y = []
    skipped = 0

    for ann in tqdm(annotations, desc="Processing videos"):
        if not ann.video_path.exists():
            logger.warning(f"Path not found, skipping: {ann.video_path}")
            skipped += 1
            continue

        is_image_seq = ann.video_path.is_dir()

        # Get total frames for labeling
        try:
            if is_image_seq:
                with ImageSequenceReader(ann.video_path) as reader:
                    total_frames = reader.metadata.total_frames
            else:
                with VideoReader(ann.video_path) as reader:
                    total_frames = reader.metadata.total_frames
        except (IOError, ValueError) as e:
            logger.warning(f"Cannot read {ann.video_path}: {e}")
            skipped += 1
            continue

        # Process through YOLO + BoT-SORT + feature extraction
        if is_image_seq:
            sequences = process_image_sequence_to_features(
                ann.video_path, detector, feature_extractor, window_size, stride
            )
        else:
            sequences = process_video_to_features(
                ann.video_path, detector, feature_extractor, window_size, stride
            )

        if len(sequences) == 0:
            logger.warning(f"No sequences extracted from {ann.video_path.name}")
            skipped += 1
            continue

        # Assign labels to each window
        labels = assign_labels(
            num_windows=len(sequences),
            annotation=ann,
            window_size=window_size,
            stride=stride,
            total_frames=total_frames,
        )

        X_video = np.array(sequences, dtype=np.float32)
        all_X.append(X_video)
        all_y.append(labels)

        logger.debug(
            f"  {ann.video_path.name}: {len(sequences)} windows, "
            f"label={ann.label}, classes={dict(zip(*np.unique(labels, return_counts=True)))}"
        )

    if not all_X:
        logger.error(f"No data extracted from any video! ({skipped} skipped)")
        empty_x = np.empty((0, window_size, NUM_FEATURES), dtype=np.float32)
        return empty_x, np.empty(0, dtype=np.int64)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Validate
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        logger.warning(f"Found {nan_count} NaN values in features, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)

    inf_mask = np.isinf(X)
    if np.any(inf_mask):
        inf_count = np.sum(inf_mask)
        logger.warning(f"Found {inf_count} Inf values in features, clipping")
        X = np.clip(X, -10.0, 10.0)

    unique, counts = np.unique(y, return_counts=True)
    logger.info(
        f"Dataset processing complete: {len(X)} total windows from "
        f"{len(annotations) - skipped}/{len(annotations)} videos. "
        f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}. "
        f"Skipped: {skipped}"
    )

    return X, y


def generate_synthetic_data(
    num_samples: int = 3000,
    window_size: int = 30,
    num_features: int = NUM_FEATURES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for fall detection.

    Creates realistic feature sequences for three classes:
        0: Normal activities (walking, standing, sitting)
        1: Falling (rapid transition)
        2: Fallen (on ground)

    Args:
        num_samples: Total number of samples to generate.
        window_size: Frames per sequence.
        num_features: Features per frame.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X is (N, window_size, num_features) and y is (N,).
    """
    rng = np.random.default_rng(seed)

    samples_per_class = num_samples // 3
    X_list = []
    y_list = []

    # --- Class 0: Normal activities ---
    for _ in range(samples_per_class):
        seq = np.zeros((window_size, num_features), dtype=np.float32)

        # Body angle: small, stable (0-0.15 normalized = 0-27 degrees)
        base_angle = rng.uniform(0.0, 0.15)
        seq[:, 0] = base_angle + rng.normal(0, 0.02, window_size)

        # CoG height: stable, in standing range (0.4-0.7)
        base_cog = rng.uniform(0.4, 0.7)
        seq[:, 1] = base_cog + rng.normal(0, 0.01, window_size)

        # Vertical velocity: near zero
        seq[:, 2] = rng.normal(0, 0.005, window_size)

        # Bbox aspect ratio: standing (0.2-0.6)
        base_ratio = rng.uniform(0.2, 0.6)
        seq[:, 3] = base_ratio + rng.normal(0, 0.02, window_size)

        # Keypoint confidence: high (0.7-1.0)
        seq[:, 4] = rng.uniform(0.7, 1.0, window_size)

        seq = np.clip(seq, 0.0, None)
        X_list.append(seq)
        y_list.append(0)

    # --- Class 1: Falling ---
    for _ in range(samples_per_class):
        seq = np.zeros((window_size, num_features), dtype=np.float32)

        # Fall happens in the middle of the window
        fall_start = rng.integers(5, 15)
        fall_duration = rng.integers(5, 15)
        fall_end = min(fall_start + fall_duration, window_size)

        for t in range(window_size):
            if t < fall_start:
                # Normal before fall
                seq[t, 0] = rng.uniform(0.0, 0.15)
                seq[t, 1] = rng.uniform(0.4, 0.6)
                seq[t, 3] = rng.uniform(0.2, 0.5)
            elif t < fall_end:
                # During fall: rapid angle increase, CoG drops
                progress = (t - fall_start) / max(fall_duration, 1)
                seq[t, 0] = 0.1 + progress * rng.uniform(0.3, 0.5)
                seq[t, 1] = 0.5 + progress * rng.uniform(0.2, 0.4)
                seq[t, 3] = 0.4 + progress * rng.uniform(0.5, 1.5)
            else:
                # After fall: on ground
                seq[t, 0] = rng.uniform(0.4, 0.6)
                seq[t, 1] = rng.uniform(0.8, 0.95)
                seq[t, 3] = rng.uniform(1.5, 3.0)

        # Velocity: compute from CoG changes
        seq[0, 2] = 0.0
        seq[1:, 2] = np.diff(seq[:, 1])

        # Confidence: dips during fall
        for t in range(window_size):
            if fall_start <= t < fall_end:
                seq[t, 4] = rng.uniform(0.4, 0.8)
            else:
                seq[t, 4] = rng.uniform(0.7, 1.0)

        seq = np.clip(seq, 0.0, None)
        X_list.append(seq)
        y_list.append(1)

    # --- Class 2: Fallen (on ground) ---
    for _ in range(num_samples - 2 * samples_per_class):
        seq = np.zeros((window_size, num_features), dtype=np.float32)

        # Body angle: high, stable (lying down)
        base_angle = rng.uniform(0.35, 0.55)
        seq[:, 0] = base_angle + rng.normal(0, 0.03, window_size)

        # CoG height: high (near ground = bottom of frame)
        base_cog = rng.uniform(0.8, 0.95)
        seq[:, 1] = base_cog + rng.normal(0, 0.01, window_size)

        # Velocity: near zero (stationary on ground)
        seq[:, 2] = rng.normal(0, 0.003, window_size)

        # Bbox aspect ratio: wide/flat
        base_ratio = rng.uniform(1.5, 3.0)
        seq[:, 3] = base_ratio + rng.normal(0, 0.1, window_size)

        # Confidence: moderate
        seq[:, 4] = rng.uniform(0.5, 0.9, window_size)

        seq = np.clip(seq, 0.0, None)
        X_list.append(seq)
        y_list.append(2)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    logger.info(
        f"Generated synthetic data: X={X.shape}, y={y.shape}, "
        f"class distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
    )
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split data into train/val/test sets with stratification.

    Args:
        X: Feature array of shape (N, window_size, num_features).
        y: Label array of shape (N,).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation. Test gets the remainder.
        seed: Random seed.

    Returns:
        Dict with 'train', 'val', 'test' keys, each mapping to (X, y) tuple.
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=seed, stratify=y
    )

    # Second split: val vs test
    val_fraction = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_fraction), random_state=seed, stratify=y_temp
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def save_splits(splits: dict, output_dir: str | Path) -> None:
    """Save train/val/test splits as .npy files.

    Args:
        splits: Dict from split_data().
        output_dir: Directory to save .npy files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (X, y) in splits.items():
        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)
        logger.info(f"Saved {split_name}: X={X.shape}, y={y.shape}")


def load_splits(data_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load train/val/test splits from .npy files.

    Args:
        data_dir: Directory containing the .npy files.

    Returns:
        Dict with 'train', 'val', 'test' keys.
    """
    data_dir = Path(data_dir)
    splits = {}

    for split_name in ["train", "val", "test"]:
        X_path = data_dir / f"X_{split_name}.npy"
        y_path = data_dir / f"y_{split_name}.npy"
        if X_path.exists() and y_path.exists():
            splits[split_name] = (np.load(X_path), np.load(y_path))
            logger.info(f"Loaded {split_name}: X={splits[split_name][0].shape}")
        else:
            logger.warning(f"Missing split files for '{split_name}' in {data_dir}")

    return splits


def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    noise_sigma: float = 0.02,
    augment_factor: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data with noise and time warping.

    Args:
        X: Feature array of shape (N, T, F).
        y: Label array of shape (N,).
        noise_sigma: Standard deviation of Gaussian noise.
        augment_factor: How many augmented copies per original sample.
        seed: Random seed.

    Returns:
        Augmented (X, y) arrays including original samples.
    """
    rng = np.random.default_rng(seed)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(augment_factor):
        # Add Gaussian noise
        noise = rng.normal(0, noise_sigma, X.shape).astype(np.float32)
        X_noisy = np.clip(X + noise, 0.0, None)
        X_aug_list.append(X_noisy)
        y_aug_list.append(y.copy())

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)

    # Shuffle
    indices = rng.permutation(len(X_aug))
    return X_aug[indices], y_aug[indices]
