import numpy as np


def sample_vertex_indices(
    num_vertices: int,
    num_samples: int | None,
    seed: int | None = None,
    points: np.ndarray | None = None,
) -> np.ndarray:
    """Return FPS vertex indices for fixed-size point-cloud samples."""
    if num_vertices <= 0:
        raise ValueError("Cannot sample from an empty mesh")
    if num_samples is None or num_samples <= 0:
        return np.arange(num_vertices, dtype=np.int64)

    return farthest_point_sample(_require_points(points), num_samples, seed=seed)


def farthest_point_sample(
    points: np.ndarray, num_samples: int, seed: int | None = None
) -> np.ndarray:
    """Sample existing vertices with farthest point sampling."""
    points = _as_points(points)
    rng = np.random.default_rng(seed)

    if num_samples >= len(points):
        return _all_indices_with_optional_padding(len(points), num_samples, rng)

    start_idx = int(rng.integers(len(points)))
    fast_indices = _farthest_point_sample_fast(points, num_samples, start_idx=start_idx)
    if fast_indices is not None:
        return fast_indices

    return _farthest_point_sample_numpy(points, num_samples, start_idx)


def _farthest_point_sample_numpy(
    points: np.ndarray, num_samples: int, start_idx: int
) -> np.ndarray:
    selected = np.empty(num_samples, dtype=np.int64)
    min_dist2 = np.full(len(points), np.inf, dtype=np.float32)
    farthest = start_idx

    for i in range(num_samples):
        selected[i] = farthest
        diff = points - points[farthest]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
        farthest = int(np.argmax(min_dist2))

    return selected


def _farthest_point_sample_fast(
    points: np.ndarray, num_samples: int, start_idx: int
) -> np.ndarray | None:
    if len(points) < 32:
        return None
    try:
        from fpsample import bucket_fps_kdline_sampling
    except ImportError:
        return None

    indices = bucket_fps_kdline_sampling(points, num_samples, h=5, start_idx=start_idx)
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) != num_samples:
        raise RuntimeError(f"FPS sampled {len(indices)} points, expected {num_samples}")
    return indices


def _require_points(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        raise ValueError("FPS sampling requires points")
    return points


def _as_points(points: np.ndarray) -> np.ndarray:
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    return points


def _all_indices_with_optional_padding(
    num_vertices: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    base = np.arange(num_vertices, dtype=np.int64)
    if num_samples == num_vertices:
        return base
    extra = rng.choice(num_vertices, size=num_samples - num_vertices, replace=True)
    return np.concatenate([base, extra.astype(np.int64)]).astype(np.int64)
