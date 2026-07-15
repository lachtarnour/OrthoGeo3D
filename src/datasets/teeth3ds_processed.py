from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.labels import (
    ARCH_CLASS_LABELS,
    FDI_LABELS,
)
from src.utils.config import load_config
from src.utils.io import load_processed_sample
from src.utils.paths import get_processed_dir, resolve_project_path


DEFAULT_FEATURE_KEYS = ("pos", "normal")
DataConfig = str | Path | Mapping[str, Any]
FEATURE_DIMS = {
    "pos": 3,
    "normal": 3,
    "curvature": 1,
}


class ProcessedScanDataset(Dataset):
    """Base loader for reusable scan-level `.pt` samples."""

    def __init__(
        self,
        split: str,
        processed_dir: str | Path | None = None,
        split_source: str = "teethseg22",
        limit: int | None = None,
    ) -> None:
        self.split = split
        self.split_source = split_source
        self.processed_dir = (
            Path(processed_dir) if processed_dir else get_processed_dir(split_source)
        )
        self.paths = sorted((self.processed_dir / split).glob("*.pt"))
        if limit is not None:
            self.paths = self.paths[: int(limit)]
        if not self.paths:
            raise FileNotFoundError(
                f"No .pt files found in {self.processed_dir / split}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return load_processed_sample(self.paths[index])


class Teeth3DSSegmentationDataset(ProcessedScanDataset):
    """Configurable segmentation view of the scan-level Teeth3DS samples."""

    def __init__(
        self,
        split: str,
        processed_dir: str | Path | None = None,
        split_source: str = "teethseg22",
        feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
        target_key: str = "y_arch_class",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(
            split=split,
            processed_dir=processed_dir,
            split_source=split_source,
            limit=limit,
        )
        self.feature_keys = tuple(feature_keys)
        self.target_key = target_key
        self.transform = transform
        unknown = set(self.feature_keys) - set(FEATURE_DIMS)
        if unknown:
            raise ValueError(f"Unsupported feature keys: {sorted(unknown)}")

    @classmethod
    def from_config(
        cls,
        config_path: DataConfig,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> "Teeth3DSSegmentationDataset":
        config = read_data_config(config_path)
        dataset_config = config.get("dataset", {})
        segmentation_config = config.get("segmentation_dataset", {})
        processed_dir = resolve_project_path(
            config.get("paths", {}).get("processed_dir")
        )
        return cls(
            split=split,
            processed_dir=processed_dir,
            split_source=str(dataset_config.get("split_source", "teethseg22")),
            feature_keys=segmentation_config.get("feature_keys", DEFAULT_FEATURE_KEYS),
            target_key=str(segmentation_config.get("target_key", "y_arch_class")),
            transform=transform,
            limit=limit,
        )

    @property
    def feature_dim(self) -> int:
        return sum(FEATURE_DIMS[key] for key in self.feature_keys)

    @property
    def num_classes(self) -> int:
        return target_num_classes(self.target_key)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return build_segmentation_item(
            super().__getitem__(index),
            feature_keys=self.feature_keys,
            target_key=self.target_key,
            transform=self.transform,
        )


def build_segmentation_item(
    sample: dict[str, Any],
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
    target_key: str = "y_arch_class",
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    item = {
        "scan_id": sample["scan_id"],
        "patient_id": sample["patient_id"],
        "jaw": sample["jaw"],
        "pos": torch.as_tensor(sample["pos"], dtype=torch.float32),
        "normal": torch.as_tensor(sample["normal"], dtype=torch.float32),
        "y_binary": torch.as_tensor(sample["y_binary"], dtype=torch.long),
        "y_fdi": torch.as_tensor(sample["y_fdi"], dtype=torch.long),
        "y_fdi_class": torch.as_tensor(sample["y_fdi_class"], dtype=torch.long),
        "y_arch_class": torch.as_tensor(sample["y_arch_class"], dtype=torch.long),
        "y_instance": torch.as_tensor(sample["y_instance"], dtype=torch.long),
    }
    add_requested_features(item, sample, feature_keys)
    if transform is not None:
        item = transform(item)
    item["x"] = build_features(item, feature_keys)
    item["y"] = item[target_key]
    validate_segmentation_item(item)
    return item


def add_requested_features(
    item: dict[str, Any],
    sample: dict[str, Any],
    feature_keys: Sequence[str],
) -> None:
    for key in feature_keys:
        if key in item:
            continue
        value = sample.get(key)
        if value is None:
            raise ValueError(
                f"Feature {key!r} is missing from sample {sample.get('scan_id')}"
            )
        item[key] = torch.as_tensor(value, dtype=torch.float32)


def build_features(
    sample: dict[str, Any],
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
) -> torch.Tensor:
    features = []
    positions = torch.as_tensor(sample["pos"])
    for key in feature_keys:
        value = sample.get(key)
        if value is None:
            raise ValueError(
                f"Feature {key!r} is missing from sample {sample.get('scan_id')}"
            )
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim == positions.ndim - 1:
            value = value.unsqueeze(-1)
        features.append(value)
    return torch.cat(features, dim=-1)


def target_num_classes(target_key: str) -> int:
    if target_key == "y_arch_class":
        return len(ARCH_CLASS_LABELS)
    if target_key == "y_binary":
        return 2
    if target_key == "y_fdi_class":
        return len(FDI_LABELS)
    raise ValueError(f"Unknown segmentation target: {target_key!r}")


def validate_segmentation_item(item: dict[str, Any]) -> None:
    pos = item["pos"]
    normal = item["normal"]
    target = item["y"]
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected pos with shape [N, 3], got {tuple(pos.shape)}")
    if normal.shape != pos.shape:
        raise ValueError(
            f"Expected normal with shape {tuple(pos.shape)}, got {tuple(normal.shape)}"
        )
    if target.shape != pos.shape[:1]:
        raise ValueError(
            f"Expected target with shape {tuple(pos.shape[:1])}, got {tuple(target.shape)}"
        )


def create_segmentation_dataloader(
    config_path: DataConfig,
    split: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    limit: int | None = None,
) -> DataLoader:
    config = read_data_config(config_path)
    dataset = Teeth3DSSegmentationDataset.from_config(
        config_path=config_path,
        split=split,
        transform=transform,
        limit=limit,
    )
    loader_config = config["dataloader"]
    mode_config = loader_config["train" if split == "train" else "eval"]
    num_workers = int(mode_config["num_workers"])
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(mode_config["batch_size"]),
        "shuffle": bool(mode_config["shuffle"]),
        "num_workers": num_workers,
        "pin_memory": bool(mode_config.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            mode_config.get("persistent_workers", False)
        )
        if "prefetch_factor" in mode_config:
            loader_kwargs["prefetch_factor"] = int(mode_config["prefetch_factor"])
    return DataLoader(dataset, **loader_kwargs)


def read_data_config(config: DataConfig) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return load_config(config)
