import unittest

import torch

from src.datasets.teeth3ds_processed import create_segmentation_dataloader
from src.training.utils import random_sample_point_batch


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

    def test_random_training_sampler_keeps_pointwise_tensors_aligned(self) -> None:
        point_ids = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
        batch = {
            "scan_id": "scan_1",
            "x": point_ids.repeat(1, 1, 3),
            "pos": point_ids.repeat(1, 1, 3),
            "normal": point_ids.repeat(1, 1, 3) + 100.0,
            "y": torch.arange(10, dtype=torch.long).view(1, 10),
            "y_fdi_class": torch.arange(10, dtype=torch.long).view(1, 10),
        }
        generator = torch.Generator().manual_seed(0)

        sampled = random_sample_point_batch(batch, num_points=4, generator=generator)

        self.assertEqual(sampled["scan_id"], "scan_1")
        self.assertEqual(sampled["x"].shape, (1, 4, 3))
        self.assertEqual(sampled["y"].shape, (1, 4))
        self.assertTrue(torch.equal(sampled["x"][0, :, 0].long(), sampled["y"][0]))
        self.assertTrue(torch.equal(sampled["pos"][0, :, 0].long(), sampled["y_fdi_class"][0]))
        self.assertTrue(torch.equal((sampled["normal"][0, :, 0] - 100.0).long(), sampled["y"][0]))


if __name__ == "__main__":
    unittest.main()
