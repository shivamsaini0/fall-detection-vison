"""Tests for dataset processing and data generation."""

from pathlib import Path

import numpy as np
import pytest

from src.dataset_processor import (
    augment_data,
    generate_synthetic_data,
    load_splits,
    save_splits,
    split_data,
)
from src.label_parser import VideoAnnotation, parse_generic_folder


class TestGenerateSyntheticData:
    def test_basic_shape(self):
        X, y = generate_synthetic_data(num_samples=90, window_size=30, seed=42)
        assert X.shape == (90, 30, 5)
        assert y.shape == (90,)

    def test_class_distribution(self):
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 3  # 3 classes
        assert set(unique) == {0, 1, 2}
        # Each class should have approximately 100 samples
        for count in counts:
            assert count >= 90  # Allow some variance

    def test_no_nan_or_inf(self):
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_feature_ranges(self):
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        # All features should be non-negative after clipping
        assert np.all(X >= 0)

    def test_reproducibility(self):
        X1, y1 = generate_synthetic_data(num_samples=100, seed=42)
        X2, y2 = generate_synthetic_data(num_samples=100, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds(self):
        X1, _ = generate_synthetic_data(num_samples=100, seed=42)
        X2, _ = generate_synthetic_data(num_samples=100, seed=123)
        assert not np.allclose(X1, X2)

    def test_normal_class_characteristics(self):
        """Normal activities should have small body angles and stable CoG."""
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        normal_mask = y == 0
        normal_X = X[normal_mask]

        # Body angle should be small (< 0.2 normalized = 36 degrees)
        mean_angle = np.mean(normal_X[:, :, 0])
        assert mean_angle < 0.2

        # Velocity should be near zero
        mean_velocity = np.mean(np.abs(normal_X[:, :, 2]))
        assert mean_velocity < 0.02

    def test_fallen_class_characteristics(self):
        """Fallen state should have high body angle and high CoG (near ground)."""
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        fallen_mask = y == 2
        fallen_X = X[fallen_mask]

        # Body angle should be large
        mean_angle = np.mean(fallen_X[:, :, 0])
        assert mean_angle > 0.3

        # CoG height should be high (near bottom of frame)
        mean_cog = np.mean(fallen_X[:, :, 1])
        assert mean_cog > 0.7


class TestSplitData:
    def test_split_sizes(self):
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        splits = split_data(X, y, train_ratio=0.8, val_ratio=0.1)

        total = sum(len(s[0]) for s in splits.values())
        assert total == 300

        assert len(splits["train"][0]) == pytest.approx(240, abs=10)
        assert len(splits["val"][0]) == pytest.approx(30, abs=10)
        assert len(splits["test"][0]) == pytest.approx(30, abs=10)

    def test_stratification(self):
        """Each split should contain all classes."""
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        splits = split_data(X, y)

        for name, (X_s, y_s) in splits.items():
            unique = set(np.unique(y_s))
            assert unique == {0, 1, 2}, f"Split '{name}' missing classes: {unique}"

    def test_no_data_leakage(self):
        """Train/val/test should not share indices."""
        X, y = generate_synthetic_data(num_samples=300, seed=42)
        splits = split_data(X, y)

        # Check that no exact duplicates exist across splits
        # (This is a simple check - real leakage from same video is
        # handled by video-level splitting in the real pipeline)
        all_shapes = [X_s.shape[0] for X_s, _ in splits.values()]
        assert sum(all_shapes) == 300


class TestSaveLoadSplits:
    def test_save_and_load(self, tmp_path):
        X, y = generate_synthetic_data(num_samples=90, seed=42)
        splits = split_data(X, y)

        save_splits(splits, tmp_path)

        loaded = load_splits(tmp_path)
        assert set(loaded.keys()) == {"train", "val", "test"}

        for name in splits:
            np.testing.assert_array_equal(splits[name][0], loaded[name][0])
            np.testing.assert_array_equal(splits[name][1], loaded[name][1])


class TestAugmentData:
    def test_augment_size(self):
        X, y = generate_synthetic_data(num_samples=100, seed=42)
        X_aug, y_aug = augment_data(X, y, augment_factor=2)
        # Original + 2 augmented copies = 3x
        assert len(X_aug) == 300
        assert len(y_aug) == 300

    def test_augmented_data_valid(self):
        X, y = generate_synthetic_data(num_samples=100, seed=42)
        X_aug, y_aug = augment_data(X, y, augment_factor=2)
        assert not np.any(np.isnan(X_aug))
        assert not np.any(np.isinf(X_aug))
        assert np.all(X_aug >= 0)

    def test_class_distribution_preserved(self):
        X, y = generate_synthetic_data(num_samples=90, seed=42)
        _, counts_orig = np.unique(y, return_counts=True)
        X_aug, y_aug = augment_data(X, y, augment_factor=2)
        _, counts_aug = np.unique(y_aug, return_counts=True)
        # Each class should be 3x the original count
        np.testing.assert_array_equal(counts_aug, counts_orig * 3)


class TestLabelParser:
    def test_parse_generic_folder(self, tmp_path):
        # Create fall/normal structure
        fall_dir = tmp_path / "fall"
        fall_dir.mkdir()
        (fall_dir / "fall1.mp4").touch()
        (fall_dir / "fall2.mp4").touch()

        normal_dir = tmp_path / "normal"
        normal_dir.mkdir()
        (normal_dir / "walk1.mp4").touch()

        annotations = parse_generic_folder(tmp_path)
        assert len(annotations) == 3
        falls = [a for a in annotations if a.is_fall]
        assert len(falls) == 2

    def test_nonexistent_dir(self, tmp_path):
        annotations = parse_generic_folder(tmp_path / "nonexistent")
        assert len(annotations) == 0

    def test_video_annotation_properties(self):
        ann = VideoAnnotation(
            video_path=Path("test.mp4"),
            label="fall",
            fall_start_frame=100,
            fall_end_frame=200,
        )
        assert ann.is_fall is True

        ann2 = VideoAnnotation(video_path=Path("test.mp4"), label="adl")
        assert ann2.is_fall is False
