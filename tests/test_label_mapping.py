import unittest

import numpy as np

from src.datasets.labels import map_arch_class_to_fdi, map_fdi_to_arch_class


class TestLabelMapping(unittest.TestCase):
    def test_fdi_to_arch_class_merges_symmetric_positions(self) -> None:
        labels = np.asarray([0, 18, 11, 21, 28, 38, 31, 41, 48], dtype=np.int64)

        mapped = map_fdi_to_arch_class(labels)

        self.assertEqual(mapped.tolist(), [0, 1, 8, 9, 16, 1, 8, 9, 16])

    def test_arch_class_to_fdi_uses_jaw(self) -> None:
        labels = np.asarray([0, 1, 8, 9, 16], dtype=np.int64)

        upper = map_arch_class_to_fdi(labels, jaw="upper")
        lower = map_arch_class_to_fdi(labels, jaw="lower")

        self.assertEqual(upper.tolist(), [0, 18, 11, 21, 28])
        self.assertEqual(lower.tolist(), [0, 38, 31, 41, 48])


if __name__ == "__main__":
    unittest.main()
