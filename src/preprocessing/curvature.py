import numpy as np


def compute_mean_curvature_magnitude(
    vertices: np.ndarray,
    faces: np.ndarray | None,
    clip_percentile: float = 99.0,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Estimate normalized mean-curvature magnitude with cotangent weights."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape [N, 3], got {vertices.shape}")
    if faces is None or len(faces) == 0 or len(vertices) == 0:
        return np.zeros(len(vertices), dtype=np.float32)

    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces with shape [F, 3], got {faces.shape}")

    tri = vertices[faces]
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(p1 - p0, p2 - p0)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > eps
    if not np.any(valid):
        return np.zeros(len(vertices), dtype=np.float32)

    faces = faces[valid]
    p0, p1, p2 = p0[valid], p1[valid], p2[valid]
    double_area = double_area[valid]
    face_area = 0.5 * double_area

    cot0 = _cotangent(p1 - p0, p2 - p0, double_area)
    cot1 = _cotangent(p0 - p1, p2 - p1, double_area)
    cot2 = _cotangent(p0 - p2, p1 - p2, double_area)

    laplace = np.zeros_like(vertices, dtype=np.float64)
    vertex_area = np.zeros(len(vertices), dtype=np.float64)

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    _accumulate_edge(laplace, j, k, p1, p2, cot0)
    _accumulate_edge(laplace, i, k, p0, p2, cot1)
    _accumulate_edge(laplace, i, j, p0, p1, cot2)

    third_area = face_area / 3.0
    np.add.at(vertex_area, i, third_area)
    np.add.at(vertex_area, j, third_area)
    np.add.at(vertex_area, k, third_area)

    valid_vertices = vertex_area > eps
    mean_curvature = np.zeros(len(vertices), dtype=np.float64)
    laplace[valid_vertices] /= 2.0 * vertex_area[valid_vertices, None]
    mean_curvature[valid_vertices] = 0.5 * np.linalg.norm(laplace[valid_vertices], axis=1)
    return _robust_unit_scale(mean_curvature, clip_percentile=clip_percentile, eps=eps)


def _cotangent(edge_a: np.ndarray, edge_b: np.ndarray, double_area: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", edge_a, edge_b) / double_area


def _accumulate_edge(
    laplace: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    src_pos: np.ndarray,
    dst_pos: np.ndarray,
    weight: np.ndarray,
) -> None:
    delta = weight[:, None] * (dst_pos - src_pos)
    np.add.at(laplace, src_idx, delta)
    np.add.at(laplace, dst_idx, -delta)


def _robust_unit_scale(values: np.ndarray, clip_percentile: float, eps: float) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    positive = values[values > 0]
    if len(positive) == 0:
        return np.zeros_like(values, dtype=np.float32)

    scale = float(np.percentile(positive, clip_percentile))
    if scale <= eps:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)
