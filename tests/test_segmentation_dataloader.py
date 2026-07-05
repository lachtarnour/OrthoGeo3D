import unittest

import torch

from src.datasets.teeth3ds_processed import create_segmentation_dataloader


class TestSegmentationDataLoader(unittest.TestCase):
    def test_train_batch(self) -> None:
        loader = create_segmentation_dataloader("configs/data.yaml", split="train", limit=2)
        batch = next(iter(loader))

        self.assertEqual(batch["x"].shape, (2, 30000, 6))
        self.assertEqual(batch["y"].shape, (2, 30000))
        self.assertEqual(batch["x"].dtype, torch.float32)
        self.assertEqual(batch["y"].dtype, torch.long)
        self.assertGreaterEqual(int(batch["y"].min()), 0)
        self.assertLess(int(batch["y"].max()), 33)


if __name__ == "__main__":
    unittest.main()
