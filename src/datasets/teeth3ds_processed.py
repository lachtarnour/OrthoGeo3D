from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.datasets.teeth3ds_raw import FDI_LABELS
from src.utils.io import load_processed_sample
from src.utils.paths import get_processed_dir, resolve_project_path
from src.utils.config import load_config


DEFAULT_FEATURE_KEYS = ("pos", "normal")
FEATURE_DIMS = {
    "pos": 3,
    "normal": 3,
    "curvature": 1,
}


class ProcessedScanDataset(Dataset):
    """Base dataset load processed scan"""
    def __init__(
        self,
        split: str,
        processed_dir: str | Path | None = None,
        limit: int | None = None
    ) -> None:
        self.split = split 
        self.processed_dir = Path(processed_dir) if processed_dir else get_processed_dir()
        self.paths = sorted((self.processed_dir / split).glob("*.pt"))
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {self.processed_dir / split}")
    
    def __len__(self)->int:
        return len(self.paths)
    
    def __getitem__(self, index):
        return self.load_sample(index)
    
    def load_sample(self,index:int)->dict[str,Any]:
        return load_processed_sample(self.paths[index])
    

class Teeth3DSSegmentationDataset(ProcessedScanDataset):
    """FDI segmentation dataset
    Input data build from 'feature_keys'
    """

    def __init__(
        self,
        split: str,
        processed_dir: str | Path | None = None,
        feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
        target_key: str = "y_fdi_class",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(split=split, processed_dir=processed_dir, limit=limit)
        self.feature_keys = tuple(feature_keys)
        self.target_key = target_key
        self.transform = transform
        unknown_features = set(self.feature_keys) - set(FEATURE_DIMS)
        if unknown_features:
            raise ValueError(f"Unsupported feature keys: {unknown_features}")
        
    @classmethod
    def from_config(
        cls,
        config_path:str | Path,
        split: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> "Teeth3DSSegmentationDataset": 
        config = load_config(config_path)
        paths_config = config.get("paths",{})
        dataset_config = config.get("segmentation_dataset",{})
        return cls(
            split=split,
            processed_dir=resolve_project_path(paths_config.get("processed_dir")),
            feature_keys=dataset_config.get("feature_keys", DEFAULT_FEATURE_KEYS),
            target_key=dataset_config.get("target_key", "y_fdi_class"),
            transform=transform,
            limit=limit,
        )
    
    @property
    def feature_dim(self) -> int:
        return sum(FEATURE_DIMS[key] for key in self.feature_keys)

    @property
    def num_classes(self) -> int:
        return len(FDI_LABELS)
    
    def __getitem__(self,index:int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        return build_segmentation_item(
            sample,
            feature_keys = self.feature_keys,
            target_key=self.target_key,
            transform=self.transform,
        )



def build_segmentation_item(
    sample: dict[str,Any],
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
    target_key: str = "y_fdi_class",
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
        "y_instance": torch.as_tensor(sample["y_instance"], dtype=torch.long),
    }
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
        value = sample.get(key)
        if value is None:
            raise ValueError(f"Feature {key!r} is missing from sample {sample.get('scan_id')}")
        item[key] = torch.as_tensor(value, dtype=torch.float32)



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



def compute_class_counts(dataset: Teeth3DSSegmentationDataset, num_classes: int = len(FDI_LABELS)) -> torch.Tensor:
    counts = torch.zeros(num_classes,dtype=torch.long)
    for path in dataset.paths:
        sample = load_processed_sample(path)
        labels = torch.as_tensor(sample[dataset.target_key], dtype=torch.long)        
        counts += torch.bincount(labels, minlength=num_classes)
    return counts

def compute_class_weights(counts: torch.Tensor, smoothing: float = 1.02) -> torch.Tensor:
    frequencies = counts.float() / counts.sum().clamp_min(1)
    weights = 1.0 / torch.log(smoothing + frequencies)
    return weights / weights.mean()


def create_segmentation_dataloader(
    config_path: str | Path,
    split: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    limit: int | None = None
) -> DataLoader:
    config = load_config(config_path)
    dataset = Teeth3DSSegmentationDataset.from_config(
        config_path=config_path,
        split=split,
        transform=transform,
        limit=limit,
    )
    loader_config = config.get("dataloader", {})
    mode_config = loader_config["train"] if split == "train" else loader_config.get("eval", {})
    return DataLoader(
        dataset,
        batch_size=int(mode_config.get("batch_size", 2)),
        shuffle=bool(mode_config.get("shuffle", split == "train")),
        num_workers=int(mode_config.get("num_workers", 0)),
    )
    
