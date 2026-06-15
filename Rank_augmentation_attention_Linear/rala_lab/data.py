"""Datasets for fast local experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class DatasetBundle:
    train: DataLoader
    val: DataLoader
    input_shape: tuple[int, int, int]
    num_classes: int


class SyntheticPatternDataset(Dataset):
    """Harder image-like dataset with overlapping class-specific patterns.

    8 classes, each defined by a pair of spatial patches at different
    positions with random noise and slight perturbations.  Much harder
    than the original 4-class corner-patch dataset.
    """

    def __init__(
        self,
        samples: int,
        image_size: int = 16,
        num_classes: int = 8,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.num_classes = num_classes
        # stronger background noise
        self.images = torch.randn(samples, 1, image_size, image_size, generator=generator) * 0.4
        self.labels = torch.randint(0, num_classes, (samples,), generator=generator)
        patch = max(2, image_size // 6)

        # define overlapping positions for 8 classes
        positions = [
            [(0, 0), (image_size // 2, image_size // 2)],
            [(0, image_size - patch), (image_size // 2, 0)],
            [(image_size - patch, 0), (0, image_size // 2)],
            [(image_size - patch, image_size - patch), (image_size // 2, image_size // 4)],
            [(image_size // 4, image_size // 4), (image_size * 3 // 4 - patch, 0)],
            [(image_size // 4, image_size * 3 // 4 - patch), (0, image_size // 4)],
            [(image_size * 3 // 4 - patch, image_size // 4), (image_size // 4, 0)],
            [(image_size // 2, 0), (0, 0)],
        ]
        # different intensities per class to create subtlety
        intensities = [0.7, 0.8, 0.9, 1.0, 0.75, 0.85, 0.95, 0.65]

        for idx, label in enumerate(self.labels.tolist()):
            cls_positions = positions[label % len(positions)]
            intensity = intensities[label % len(intensities)]
            for row, col in cls_positions:
                r_end = min(row + patch, image_size)
                c_end = min(col + patch, image_size)
                noise = torch.randn(1, r_end - row, c_end - col, generator=generator) * 0.15
                self.images[idx, :, row:r_end, col:c_end] += intensity + noise

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def make_dataset(
    name: str,
    batch_size: int,
    seed: int,
    limit: int = 1200,
    data_dir: str = "data",
) -> DatasetBundle:
    name = name.lower()
    if name == "synthetic":
        train_ds = SyntheticPatternDataset(limit, seed=seed)
        val_ds = SyntheticPatternDataset(max(200, limit // 5), seed=seed + 1)
        return DatasetBundle(
            train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            val=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            input_shape=(1, 16, 16),
            num_classes=8,
        )

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for MNIST, Fashion-MNIST, or CIFAR-10.") from exc

    if name in {"mnist", "fashion-mnist"}:
        train_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        dataset_cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        full_train = dataset_cls(data_dir, train=True, download=True, transform=train_transform)
        full_val = dataset_cls(data_dir, train=False, download=True, transform=val_transform)
        input_shape = (1, 28, 28)
        classes = 10
    elif name == "cifar-10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        full_val = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_transform)
        input_shape = (3, 32, 32)
        classes = 10
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_count = min(limit, len(full_train))
    val_count = min(max(200, limit // 5), len(full_val))
    generator = torch.Generator().manual_seed(seed)
    train_subset, _ = random_split(full_train, [train_count, len(full_train) - train_count], generator=generator)
    val_subset, _ = random_split(full_val, [val_count, len(full_val) - val_count], generator=generator)
    return DatasetBundle(
        train=DataLoader(train_subset, batch_size=batch_size, shuffle=True),
        val=DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        input_shape=input_shape,
        num_classes=classes,
    )
