import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps":
        has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("MPS was requested but is not available")
    return resolved


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def random_sample_point_batch(
    batch: Any,
    num_points: int | None,
    generator: torch.Generator | None = None,
) -> Any:
    """Randomly sample point-wise tensors from a collated segmentation batch."""
    if num_points is None or num_points <= 0:
        return batch
    if not isinstance(batch, Mapping) or not torch.is_tensor(batch.get("x")):
        return batch

    x = batch["x"]
    if x.ndim < 3:
        return batch

    batch_size, num_vertices = int(x.shape[0]), int(x.shape[1])
    if num_vertices <= 0:
        return batch

    num_points = int(num_points)
    indices = []
    for _ in range(batch_size):
        if num_points <= num_vertices:
            index = torch.randperm(num_vertices, generator=generator)[:num_points]
        else:
            index = torch.randint(num_vertices, (num_points,), generator=generator)
        indices.append(index)

    sampled = dict(batch)
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == batch_size and value.shape[1] == num_vertices:
            sampled[key] = torch.stack(
                [
                    value[batch_index].index_select(0, indices[batch_index].to(value.device))
                    for batch_index in range(batch_size)
                ],
                dim=0,
            )
    return sampled


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep scalar numeric metrics in a plain dictionary."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            flat[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, np.number)):
            flat[key] = float(value)
    return flat


def average_metric_dicts(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not items:
        return {}

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}
