from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Return unit vectors, keeping zero vectors stable."""
    vectors = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(lengths, eps)).astype(np.float32)


def compute_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray | None
) -> np.ndarray:
    """Compute vertex normals by averaging neighboring unit face normals."""
    vertices = np.asarray(vertices, dtype=np.float32)
    if faces is None or len(faces) == 0:
        return np.zeros_like(vertices, dtype=np.float32)

    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices, dtype=np.float32)

    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(lengths, 1.0e-12)

    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)

    return normalize_vectors(normals)
