from __future__ import annotations

import numpy as np


def sample_vertex_indices(
    num_vertices: int,
    num_samples: int | None,
    method: str = "random",
    seed: int | None = None,
) -> np.ndarray:
    """Return vertex indices for fixed-size point-cloud training samples."""
    if num_vertices <= 0:
        raise ValueError("Cannot sample from an empty mesh")
    if num_samples is None or num_samples <= 0:
        return np.arange(num_vertices, dtype=np.int64)

    rng = np.random.default_rng(seed)
    method = method.lower()

    if method == "random":
        replace = num_samples > num_vertices
        return rng.choice(num_vertices, size=num_samples, replace=replace).astype(np.int64)

    if method == "stride":
        if num_samples >= num_vertices:
            extra = rng.choice(num_vertices, size=num_samples - num_vertices, replace=True)
            return np.concatenate([np.arange(num_vertices), extra]).astype(np.int64)
        positions = np.linspace(0, num_vertices - 1, num_samples)
        return np.round(positions).astype(np.int64)

    raise ValueError(f"Unknown sampling method: {method}")
