from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.labels import ARCH_CLASS_LABELS, FDI_LABELS, map_fdi_to_arch_class
from src.utils.config import load_config
from src.utils.io import load_processed_sample
from src.utils.paths import get_processed_dir, resolve_project_path


DEFAULT_FEATURE_KEYS = ("pos", "normal")
FEATURE_DIMS = {
    "pos": 3,
    "normal": 3,
    "jaw_code": 1,
}
JAW_CODES = {
    "lower": 0.0,
    "upper": 1.0,
}


class ProcessedScanDataset(Dataset):
    """Base dataset for processed scan-level `.pt` files."""

    def __init__(
        self,
        split: str,
        processed_dir: str | Path | None = None,
        split_source: str | None = None,
        limit: int | None = None,
    ) -> None:
        self.split = split
        self.split_source = split_source
        self.processed_dir = Path(processed_dir) if processed_dir else get_processed_dir(split_source)
        self.paths = sorted((self.processed_dir / split).glob("*.pt"))
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {self.processed_dir / split}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.load_sample(index)

    def load_sample(self, index: int) -> dict[str, Any]:
        return load_processed_sample(self.paths[index])


class Teeth3DSSegmentationDataset(ProcessedScanDataset):
    """Point-wise tooth segmentation dataset built from processed Teeth3DS scans."""

    def __init__(
        self,
        split: str,
        processed_dir: str | Path | None = None,
        split_source: str | None = None,
        feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
        target_key: str = "y_arch_class",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(split=split, processed_dir=processed_dir, split_source=split_source, limit=limit)
        self.feature_keys = tuple(feature_keys)
        self.target_key = target_key
        self.transform = transform
        unknown_features = set(self.feature_keys) - set(FEATURE_DIMS)
        if unknown_features:
            raise ValueError(f"Unsupported feature keys: {sorted(unknown_features)}")

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> "Teeth3DSSegmentationDataset":
        config = load_config(config_path)
        paths_config = config.get("paths", {})
        split_source = config.get("dataset", {}).get("split_source")
        dataset_config = config.get("segmentation_dataset", {})
        return cls(
            split=split,
            processed_dir=resolve_project_path(paths_config.get("processed_dir")),
            split_source=split_source,
            feature_keys=dataset_config.get("feature_keys", DEFAULT_FEATURE_KEYS),
            target_key=dataset_config.get("target_key", "y_arch_class"),
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
        sample = super().__getitem__(index)
        return build_segmentation_item(
            sample,
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
        "y_fdi": torch.as_tensor(sample["y_fdi"], dtype=torch.long),
        "y_fdi_class": torch.as_tensor(sample["y_fdi_class"], dtype=torch.long),
        "y_binary": torch.as_tensor(sample["y_binary"], dtype=torch.long),
        "y_instance": torch.as_tensor(sample["y_instance"], dtype=torch.long),
    }
    item["y_arch_class"] = target_tensor_from_sample(sample, "y_arch_class")
    add_requested_features(item, sample, feature_keys)
    if transform is not None:
        item = transform(item)
    item["x"] = build_features(item, feature_keys)
    if target_key not in item:
        raise ValueError(f"Target {target_key!r} is missing from sample {sample.get('scan_id')}")
    item["y"] = item[target_key]
    return item


def add_requested_features(item: dict[str, Any], sample: dict[str, Any], feature_keys: Sequence[str]) -> None:
    for key in feature_keys:
        if key in item:
            continue
        if key == "jaw_code":
            item[key] = jaw_code_tensor(sample.get("jaw"), num_points=item["pos"].shape[0], scan_id=sample.get("scan_id"))
            continue
        value = sample.get(key)
        if value is None:
            raise ValueError(f"Feature {key!r} is missing from sample {sample.get('scan_id')}")
        item[key] = torch.as_tensor(value, dtype=torch.float32)


def jaw_code_tensor(jaw: Any, num_points: int, scan_id: str | None = None) -> torch.Tensor:
    jaw_key = str(jaw).lower()
    if jaw_key not in JAW_CODES:
        where = f" in sample {scan_id}" if scan_id else ""
        raise ValueError(f"Unsupported jaw {jaw!r}{where}; expected one of {sorted(JAW_CODES)}")
    return torch.full((num_points, 1), JAW_CODES[jaw_key], dtype=torch.float32)


def build_features(sample: dict[str, Any], feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS) -> torch.Tensor:
    features = []
    for key in feature_keys:
        value = sample.get(key)
        if value is None:
            raise ValueError(f"Feature {key!r} is missing from sample {sample.get('scan_id')}")
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        features.append(value)
    return torch.cat(features, dim=-1)


def target_tensor_from_sample(sample: dict[str, Any], target_key: str) -> torch.Tensor:
    if target_key == "y_arch_class" and target_key not in sample:
        y_fdi = torch.as_tensor(sample["y_fdi"], dtype=torch.long)
        arch_labels = map_fdi_to_arch_class(y_fdi.cpu().numpy())
        return torch.as_tensor(arch_labels, dtype=torch.long)
    if target_key == "y_binary" and target_key not in sample:
        y_fdi = torch.as_tensor(sample["y_fdi"], dtype=torch.long)
        return (y_fdi > 0).long()
    if target_key not in sample:
        raise ValueError(f"Target {target_key!r} is missing from sample {sample.get('scan_id')}")
    return torch.as_tensor(sample[target_key], dtype=torch.long)


def target_num_classes(target_key: str) -> int:
    if target_key == "y_arch_class":
        return len(ARCH_CLASS_LABELS)
    if target_key == "y_binary":
        return 2
    if target_key == "y_fdi_class":
        return len(FDI_LABELS)
    raise ValueError(f"Unknown segmentation target: {target_key!r}")


def create_segmentation_dataloader(
    config_path: str | Path,
    split: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    limit: int | None = None,
    dataloader_config_override: dict[str, Any] | None = None,
) -> DataLoader:
    config = load_config(config_path)
    dataset = Teeth3DSSegmentationDataset.from_config(
        config_path=config_path,
        split=split,
        transform=transform,
        limit=limit,
    )
    loader_config = config.get("dataloader", {})
    if dataloader_config_override:
        loader_config = _merge_dataloader_config(loader_config, dataloader_config_override)
    mode_config = loader_config["train"] if split == "train" else loader_config.get("eval", {})
    num_workers = int(mode_config.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(mode_config.get("batch_size", 2)),
        "shuffle": bool(mode_config.get("shuffle", split == "train")),
        "num_workers": num_workers,
        "pin_memory": bool(mode_config.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(mode_config.get("persistent_workers", False))
        if "prefetch_factor" in mode_config:
            loader_kwargs["prefetch_factor"] = int(mode_config["prefetch_factor"])

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def _merge_dataloader_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
