from pathlib import Path

import torch

from src.utils.io import save_processed_sample


def write_segmentation_fixture(root: Path, split_source: str = "teethseg22", samples: int = 2, points: int = 128) -> Path:
    processed_dir = root / "processed" / split_source / "train"
    processed_dir.mkdir(parents=True)

    for sample_idx in range(samples):
        labels = torch.arange(points, dtype=torch.long) % 17
        sample = {
            "scan_id": f"scan_{sample_idx}",
            "patient_id": f"patient_{sample_idx}",
            "jaw": "upper",
            "pos": torch.randn(points, 3),
            "normal": torch.randn(points, 3),
            "y_fdi": labels,
            "y_fdi_class": labels,
            "y_arch_class": labels,
            "y_binary": (labels > 0).long(),
            "y_instance": labels,
        }
        save_processed_sample(sample, processed_dir / f"scan_{sample_idx}.pt")

    config_path = root / "data.yaml"
    config_path.write_text(
        f"""
paths:
  processed_dir: null
dataset:
  split_source: {split_source}
segmentation_dataset:
  feature_keys: [pos, normal]
  target_key: y_arch_class
dataloader:
  train:
    batch_size: 2
    num_workers: 0
    shuffle: false
  eval:
    batch_size: 2
    num_workers: 0
    shuffle: false
""",
        encoding="utf-8",
    )
    return config_path
