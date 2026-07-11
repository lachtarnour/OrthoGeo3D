from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing.sampling import sample_vertex_indices


class PreprocessingSamplingTest(unittest.TestCase):
    def test_fps_is_deterministic_for_same_seed(self) -> None:
        points = np.stack(
            [
                np.linspace(0.0, 1.0, 30),
                np.zeros(30),
                np.zeros(30),
            ],
            axis=1,
        ).astype(np.float32)

        first = sample_vertex_indices(len(points), 10, seed=7, points=points)
        second = sample_vertex_indices(len(points), 10, seed=7, points=points)

        self.assertEqual(len(first), 10)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(np.unique(first)), 10)
        self.assertTrue(np.all((first >= 0) & (first < len(points))))

    def test_fps_requires_geometry_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires points"):
            sample_vertex_indices(10, 4, seed=7)

    def test_padding_keeps_existing_vertices_when_target_is_larger(self) -> None:
        points = np.eye(4, 3, dtype=np.float32)

        indices = sample_vertex_indices(len(points), 7, seed=11, points=points)

        self.assertEqual(len(indices), 7)
        self.assertEqual(indices[:4].tolist(), [0, 1, 2, 3])
        self.assertTrue(np.all((indices >= 0) & (indices < len(points))))


if __name__ == "__main__":
    unittest.main()
