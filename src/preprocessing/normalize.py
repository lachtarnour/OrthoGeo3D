from __future__ import annotations

from typing import Any

import numpy as np


def normalize_points(
    points: np.ndarray, eps: float = 1.0e-8
) -> tuple[np.ndarray, np.ndarray, float]:
    """Center and scale points into a unit-radius coordinate frame."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")

    center = points.mean(axis=0).astype(np.float32)
    centered = points - center
    scale = float(np.max(np.linalg.norm(centered, axis=1))) if len(points) else 1.0
    if scale < eps:
        scale = 1.0
    normalized = (centered / scale).astype(np.float32)
    return normalized, center, scale


def normalize_landmarks(
    landmarks: list[dict[str, Any]],
    center: np.ndarray,
    scale: float,
) -> list[dict[str, Any]]:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    scale = float(scale) if scale else 1.0

    normalized = []
    for lm in landmarks:
        item = dict(lm)
        item["coord_norm"] = (
            (np.asarray(item["coord"], dtype=np.float32) - center) / scale
        ).astype(np.float32)
        nearest_distance = item.get("nearest_distance")
        if nearest_distance is not None:
            item["nearest_distance_norm"] = float(nearest_distance) / scale
        normalized.append(item)
    return normalized


def normalize_tooth_centers(
    tooth_centers: dict[str, list[float]],
    center: np.ndarray,
    scale: float,
) -> dict[str, list[float]]:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    scale = float(scale) if scale else 1.0

    return {
        tooth_key: ((np.asarray(tooth_center, dtype=np.float32) - center) / scale)
        .astype(np.float32)
        .tolist()
        for tooth_key, tooth_center in tooth_centers.items()
    }
