from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from src.utils.io import load_json


LANDMARK_CLASSES = {
    "Mesial",
    "Distal",
    "Cusp",
    "InnerPoint",
    "OuterPoint",
    "FacialPoint",
}


def load_landmarks(path: str | Path | None) -> list[dict[str, Any]]:
    """Load a 3DTeethLand `*__kpt.json` file.

    The downloaded landmark files use:
    top-level keys: `version`, `description`, `key`, `objects`
    object keys: `key`, `class`, `coord`
    """
    if path is None:
        return []

    data = load_json(path)
    if set(data) != {"version", "description", "key", "objects"}:
        raise ValueError(
            f"Unexpected 3DTeethLand landmark keys in {path}: {sorted(data)}"
        )

    landmarks = []
    for item in data["objects"]:
        keys = set(item)
        if keys != {"key", "class", "coord"}:
            raise ValueError(
                f"Unexpected landmark object keys in {path}: {sorted(keys)}"
            )

        label = str(item["class"])
        if label not in LANDMARK_CLASSES:
            raise ValueError(f"Unexpected landmark class in {path}: {label}")

        coord = item["coord"]
        coord_array = np.asarray(coord, dtype=np.float32).reshape(-1)
        if coord_array.shape[0] != 3:
            raise ValueError(f"Invalid landmark coordinate in {path}: {coord}")

        landmarks.append(
            {
                "id": str(item["key"]),
                "class": label,
                "coord": coord_array,
            }
        )
    return landmarks


def associate_landmarks_to_vertices(
    landmarks: list[dict[str, Any]],
    vertices_raw: np.ndarray,
    labels: np.ndarray | None = None,
    instances: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if not landmarks:
        return []

    vertices_raw = np.asarray(vertices_raw, dtype=np.float32)
    coords = np.stack([lm["coord"] for lm in landmarks], axis=0)
    dist, idx = cKDTree(vertices_raw).query(coords, k=1)

    enriched = []
    for lm, nearest, distance in zip(landmarks, idx, dist):
        item = dict(lm)
        nearest = int(nearest)
        item["nearest_vertex"] = nearest
        item["nearest_distance"] = float(distance)
        if labels is not None:
            item["fdi"] = int(labels[nearest])
        if instances is not None:
            item["instance"] = int(instances[nearest])
        enriched.append(item)
    return enriched


def remap_landmarks_to_sample(
    landmarks: list[dict[str, Any]],
    source_indices: np.ndarray,
) -> list[dict[str, Any]]:
    index_map = {}
    for sampled_idx, source_idx in enumerate(
        np.asarray(source_indices, dtype=np.int64)
    ):
        index_map.setdefault(int(source_idx), int(sampled_idx))

    remapped = []
    for lm in landmarks:
        item = dict(lm)
        nearest = item.get("nearest_vertex")
        item["sampled_index"] = (
            index_map.get(int(nearest)) if nearest is not None else None
        )
        remapped.append(item)
    return remapped


def landmarks_to_nested_dict(
    landmarks: list[dict[str, Any]],
    coord_key: str = "coord",
    distance_key: str = "nearest_distance",
    distance_output_key: str = "nearest_distance",
) -> dict[str, Any] | None:
    if not landmarks:
        return None

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unassigned: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lm in landmarks:
        nearest_distance = lm.get(distance_key)
        record = {
            "id": lm.get("id"),
            "coord": np.asarray(lm[coord_key], dtype=np.float32).tolist(),
            "nearest_vertex": lm.get("nearest_vertex"),
            "sampled_index": lm.get("sampled_index"),
            distance_output_key: float(nearest_distance)
            if nearest_distance is not None
            else None,
        }
        cls = str(lm["class"])
        fdi = lm.get("fdi")
        if fdi is None or int(fdi) <= 0:
            unassigned[cls].append(record)
        else:
            grouped[str(int(fdi))][cls].append(record)

    result: dict[str, Any] = {"by_tooth": {k: dict(v) for k, v in grouped.items()}}
    if unassigned:
        result["unassigned"] = dict(unassigned)
    return result


def build_landmark_tooth_records(
    landmarks: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not landmarks:
        return None

    records = []
    for lm in landmarks:
        fdi = lm.get("fdi")
        instance = lm.get("instance")
        records.append(
            {
                "id": lm.get("id"),
                "class": lm.get("class"),
                "fdi": int(fdi) if fdi is not None else None,
                "instance": int(instance) if instance is not None else None,
                "tooth_key": str(int(fdi))
                if fdi is not None and int(fdi) > 0
                else None,
                "nearest_vertex": lm.get("nearest_vertex"),
                "sampled_index": lm.get("sampled_index"),
                "nearest_distance": lm.get("nearest_distance"),
            }
        )
    return records


def compute_tooth_centers(
    pos_raw: np.ndarray,
    y_fdi: np.ndarray,
) -> dict[str, list[float]]:
    pos_raw = np.asarray(pos_raw, dtype=np.float32)
    y_fdi = np.asarray(y_fdi).reshape(-1)
    if len(pos_raw) != len(y_fdi):
        raise ValueError("pos_raw and y_fdi must have the same length")

    centers: dict[str, list[float]] = {}
    for fdi in sorted(np.unique(y_fdi).tolist()):
        if int(fdi) <= 0:
            continue
        mask = y_fdi == fdi
        centers[str(int(fdi))] = pos_raw[mask].mean(axis=0).tolist()
    return centers
