import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from helpers import write_segmentation_fixture
from scripts.train_segmentation import build_scheduler, validate_model_dataset_contract
from src.datasets.teeth3ds_processed import Teeth3DSSegmentationDataset


class TestTrainSegmentationConfig(unittest.TestCase):
    def test_build_scheduler_can_disable_scheduler(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        self.assertIsNone(build_scheduler({}, optimizer))

    def test_build_scheduler_creates_cosine_scheduler(self) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        scheduler = build_scheduler(
            {
                "name": "cosine",
                "min_lr": 0.00005,
            },
            optimizer,
            max_epochs=100,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.eta_min, 0.00005)

    def test_model_dataset_contract_checks_feature_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = write_segmentation_fixture(root, samples=1)
            with patch.dict("os.environ", {"DATA_DIR": tmpdir}, clear=False):
                dataset = Teeth3DSSegmentationDataset.from_config(config_path, split="train")

        validate_model_dataset_contract({"input_channels": 6, "num_classes": 17}, dataset)
        with self.assertRaisesRegex(ValueError, "input_channels"):
            validate_model_dataset_contract({"input_channels": 7, "num_classes": 17}, dataset)


if __name__ == "__main__":
    unittest.main()
