import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from helpers import write_segmentation_fixture
from src.datasets.teeth3ds_processed import build_segmentation_item, create_segmentation_dataloader
from src.training.sampling import multiview_point_indices
from src.training.utils import overlapping_multiview_sample_point_batch, random_sample_point_batch


class TestSegmentationDataLoader(unittest.TestCase):
    def test_train_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = write_segmentation_fixture(root, samples=2)
            with patch.dict("os.environ", {"DATA_DIR": tmpdir}, clear=False):
                loader = create_segmentation_dataloader(config_path, split="train", limit=2)
                batch = next(iter(loader))

        self.assertEqual(batch["x"].shape[0], 2)
        self.assertEqual(batch["x"].shape[-1], 6)
        self.assertEqual(batch["y"].shape, batch["x"].shape[:2])
        self.assertEqual(batch["x"].dtype, torch.float32)
        self.assertEqual(batch["y"].dtype, torch.long)
        self.assertNotIn("jaw_code", batch)
        self.assertGreaterEqual(int(batch["y"].min()), 0)
        self.assertLess(int(batch["y"].max()), 17)
        self.assertIn("y_arch_class", batch)
        self.assertTrue(torch.equal(batch["y"], batch["y_arch_class"]))

    def test_jaw_code_maps_upper_and_lower_jaws(self) -> None:
        base_sample = {
            "scan_id": "scan_1",
            "patient_id": "patient_1",
            "pos": torch.zeros(4, 3),
            "normal": torch.ones(4, 3),
            "y_fdi": torch.zeros(4, dtype=torch.long),
            "y_fdi_class": torch.zeros(4, dtype=torch.long),
            "y_arch_class": torch.zeros(4, dtype=torch.long),
            "y_binary": torch.zeros(4, dtype=torch.long),
            "y_instance": torch.zeros(4, dtype=torch.long),
        }

        lower = build_segmentation_item({**base_sample, "jaw": "lower"}, feature_keys=("pos", "normal", "jaw_code"))
        upper = build_segmentation_item({**base_sample, "jaw": "upper"}, feature_keys=("pos", "normal", "jaw_code"))

        self.assertTrue(torch.all(lower["jaw_code"] == 0.0))
        self.assertTrue(torch.all(upper["jaw_code"] == 1.0))
        self.assertTrue(torch.all(lower["x"][..., -1] == 0.0))
        self.assertTrue(torch.all(upper["x"][..., -1] == 1.0))

    def test_random_training_sampler_keeps_pointwise_tensors_aligned(self) -> None:
        point_ids = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
        batch = {
            "scan_id": "scan_1",
            "x": point_ids.repeat(1, 1, 3),
            "pos": point_ids.repeat(1, 1, 3),
            "normal": point_ids.repeat(1, 1, 3) + 100.0,
            "y": torch.arange(10, dtype=torch.long).view(1, 10),
            "y_fdi_class": torch.arange(10, dtype=torch.long).view(1, 10),
            "y_arch_class": torch.arange(10, dtype=torch.long).view(1, 10),
        }
        generator = torch.Generator().manual_seed(0)

        sampled = random_sample_point_batch(batch, num_points=4, generator=generator)

        self.assertEqual(sampled["scan_id"], "scan_1")
        self.assertEqual(sampled["x"].shape, (1, 4, 3))
        self.assertEqual(sampled["y"].shape, (1, 4))
        self.assertTrue(torch.equal(sampled["x"][0, :, 0].long(), sampled["y"][0]))
        self.assertTrue(torch.equal(sampled["pos"][0, :, 0].long(), sampled["y_fdi_class"][0]))
        self.assertTrue(torch.equal(sampled["pos"][0, :, 0].long(), sampled["y_arch_class"][0]))
        self.assertTrue(torch.equal((sampled["normal"][0, :, 0] - 100.0).long(), sampled["y"][0]))

    def test_multiview_sampler_keeps_fifteen_thousand_points_from_sixty_thousand(self) -> None:
        num_vertices = 60000
        point_ids = torch.arange(num_vertices, dtype=torch.float32).view(1, num_vertices, 1)
        batch = {
            "scan_id": ["scan_1"],
            "x": point_ids.repeat(1, 1, 6),
            "pos": point_ids.repeat(1, 1, 3),
            "normal": point_ids.repeat(1, 1, 3) + 100.0,
            "y": torch.arange(num_vertices, dtype=torch.long).view(1, num_vertices),
            "y_arch_class": torch.arange(num_vertices, dtype=torch.long).view(1, num_vertices),
        }

        sampled = overlapping_multiview_sample_point_batch(
            batch,
            num_points=15000,
            core_points=10000,
            view_id=3,
            num_views=10,
            seed=42,
        )

        self.assertEqual(sampled["x"].shape, (1, 15000, 6))
        self.assertEqual(sampled["y"].shape, (1, 15000))
        self.assertTrue(torch.equal(sampled["x"][0, :, 0].long(), sampled["y"][0]))
        self.assertTrue(torch.equal(sampled["pos"][0, :, 0].long(), sampled["y_arch_class"][0]))

    def test_ten_fifteen_thousand_point_views_cover_sixty_thousand_points(self) -> None:
        views = [
            multiview_point_indices(
                num_vertices=60000,
                num_points=15000,
                core_points=10000,
                view_id=view_id,
                num_views=10,
                seed=42,
                scan_id="scan_1",
            )
            for view_id in range(10)
        ]

        covered = torch.cat(views).unique()

        self.assertEqual(covered.numel(), 60000)


if __name__ == "__main__":
    unittest.main()
