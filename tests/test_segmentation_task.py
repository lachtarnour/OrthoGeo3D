import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from helpers import write_segmentation_fixture
from src.datasets.teeth3ds_processed import create_segmentation_dataloader
from src.models.dgcnn import DGCNNSegmentation
from src.training.tasks import SegmentationTask


class TestSegmentationTask(unittest.TestCase):
    def test_step_on_real_dataloader_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = write_segmentation_fixture(root, samples=1, points=128)
            with patch.dict("os.environ", {"DATA_DIR": tmpdir}, clear=False):
                loader = create_segmentation_dataloader(config_path, split="train", limit=1)
                batch = next(iter(loader))
        batch["x"] = batch["x"][:, :64, :]
        batch["y"] = batch["y"][:, :64]
        model = DGCNNSegmentation(input_channels=6, num_classes=17, k=4, emb_dims=64, dropout=0.0)
        task = SegmentationTask(num_classes=17)

        output = task.training_step(model, batch)

        self.assertEqual(output["logits"].shape, (1, 64, 17))
        self.assertEqual(output["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIn("accuracy", output["metrics"])
        self.assertIn("miou", output["metrics"])
        self.assertIn("loss", output["metrics"])

    def test_composite_loss_reports_all_components(self) -> None:
        logits = torch.randn(2, 12, 17, requires_grad=True)
        target = torch.randint(0, 17, (2, 12))
        task = SegmentationTask(
            num_classes=17,
            loss_config={
                "components": {
                    "cross_entropy": {"enabled": True, "weight": 1.0, "max_loss": 6.0},
                    "dice": {"enabled": True, "weight": 1.0, "include_background": False},
                    "binary": {"enabled": True, "weight": 0.5, "max_loss": 6.0},
                }
            },
        )

        loss, metrics = task.loss_fn(logits, target)

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_cross_entropy", metrics)
        self.assertIn("loss_dice", metrics)
        self.assertIn("loss_binary", metrics)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
