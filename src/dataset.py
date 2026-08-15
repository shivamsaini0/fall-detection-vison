"""PyTorch Dataset and DataLoader for fall detection training data."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FallDataset(Dataset):
    """PyTorch Dataset for fall detection feature sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_splits(data_dir: str | Path) -> dict[str, FallDataset]:
    """Load train/val/test splits as FallDataset objects."""
    data_dir = Path(data_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        X = np.load(data_dir / f"X_{name}.npy")
        y = np.load(data_dir / f"y_{name}.npy")
        splits[name] = FallDataset(X, y)
    return splits


def create_dataloaders(
    splits: dict[str, FallDataset],
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict[str, DataLoader]:
    """Create DataLoaders from dataset splits."""
    return {
        "train": DataLoader(
            splits["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers,
        ),
        "val": DataLoader(
            splits["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ),
        "test": DataLoader(
            splits["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers,
        ),
    }
