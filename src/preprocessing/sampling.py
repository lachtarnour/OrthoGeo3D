import numpy as np


def sample_vertex_indices(
    num_vertices: int,
    num_samples: int | None,
    method: str = "fps",
    seed: int | None = None,
    points: np.ndarray | None = None,
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

    if method == "first":
        if num_samples <= num_vertices:
            return np.arange(num_samples, dtype=np.int64)
        extra = rng.choice(num_vertices, size=num_samples - num_vertices, replace=True)
        return np.concatenate([np.arange(num_vertices), extra]).astype(np.int64)

    if method == "fps":
        if points is None:
            raise ValueError("FPS sampling requires points")
        return farthest_point_sample(points, num_samples, seed=seed)

    raise ValueError(f"Unknown sampling method: {method}")


def farthest_point_sample(points: np.ndarray, num_samples: int, seed: int | None = None) -> np.ndarray:
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")

    num_vertices = len(points)
    rng = np.random.default_rng(seed)
    if num_samples >= num_vertices:
        extra = rng.choice(num_vertices, size=num_samples - num_vertices, replace=True)
        return np.concatenate([np.arange(num_vertices, dtype=np.int64), extra.astype(np.int64)])

    fast_indices = _farthest_point_sample_fast(points, num_samples)
    if fast_indices is not None:
        return fast_indices

    selected = np.empty(num_samples, dtype=np.int64)
    min_dist2 = np.full(num_vertices, np.inf, dtype=np.float32)
    farthest = int(rng.integers(num_vertices))

    for i in range(num_samples):
        selected[i] = farthest
        diff = points - points[farthest]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
        farthest = int(np.argmax(min_dist2))

    return selected


def _farthest_point_sample_fast(points: np.ndarray, num_samples: int) -> np.ndarray | None:
    try:
        from fpsample import bucket_fps_kdline_sampling
    except ImportError:
        return None

    indices = bucket_fps_kdline_sampling(points, num_samples, h=5, start_idx=0)
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) != num_samples:
        raise RuntimeError(f"FPS sampled {len(indices)} points, expected {num_samples}")
    return indices
