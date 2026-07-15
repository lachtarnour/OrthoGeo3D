import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve an explicit device or choose the best available one."""
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps":
        has_mps = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        if not has_mps:
            raise RuntimeError("MPS was requested but is not available")
    return resolved


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors inside common Python containers to a device."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


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


def average_metric_dicts(
    items: Sequence[Mapping[str, float]],
    weights: Sequence[float],
) -> dict[str, float]:
    if not items:
        return {}
    if len(items) != len(weights):
        raise ValueError("Metric items and weights must have the same length")

    totals: dict[str, float] = {}
    weight_sums: dict[str, float] = {}
    for item, weight in zip(items, weights, strict=True):
        for key, value in item.items():
            totals[key] = totals.get(key, 0.0) + float(value) * float(weight)
            weight_sums[key] = weight_sums.get(key, 0.0) + float(weight)
    return {key: totals[key] / weight_sums[key] for key in totals}
