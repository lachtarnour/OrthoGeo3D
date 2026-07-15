import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_mesh(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() != ".obj":
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Only .obj files are supported."
        )

    vertices, vertex_normals, faces = [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vn "):
                parts = line.split()
                vertex_normals.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            elif line.startswith("f "):
                raw = line.split()[1:]
                polygon = [int(token.split("/", 1)[0]) - 1 for token in raw]
                if len(polygon) >= 3:
                    for i in range(1, len(polygon) - 1):
                        faces.append([polygon[0], polygon[i], polygon[i + 1]])

    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)
    normals_array = None
    if len(vertex_normals) == len(vertices):
        normals_array = np.asarray(vertex_normals, dtype=np.float32)

    return {"vertices": vertices_array, "faces": faces_array, "normals": normals_array}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_teeth3ds_annotation(path: str | Path) -> dict[str, Any]:
    """Load a Teeth3DS annotation JSON file."""
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    keys = set(data)
    if keys != {"id_patient", "jaw", "labels", "instances"}:
        raise ValueError(
            f"Unexpected Teeth3DS annotation keys in {path}: {sorted(keys)}"
        )

    return {
        "id_patient": str(data["id_patient"]),
        "jaw": str(data["jaw"]),
        "labels": np.asarray(data["labels"], dtype=np.int64),
        "instances": np.asarray(data["instances"], dtype=np.int64),
    }


def _tensorize_sample(value: Any) -> Any:
    import torch

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"i", "u", "b"}:
            return torch.as_tensor(value, dtype=torch.long)
        return torch.as_tensor(value, dtype=torch.float32)
    if isinstance(value, dict):
        return {k: _tensorize_sample(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensorize_sample(v) for v in value]
    return value


def save_processed_sample(sample: dict[str, Any], path: str | Path) -> None:
    """Save a processed sample as a PyTorch .pt file."""
    path = Path(path)
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Processed samples are saved as .pt files, got: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(_tensorize_sample(sample), path)


def load_processed_sample(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Processed samples are loaded from .pt files, got: {path}")

    import torch

    return torch.load(path, map_location="cpu", weights_only=False)
