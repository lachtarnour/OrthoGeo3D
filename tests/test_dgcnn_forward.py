import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from helpers import write_segmentation_fixture
from src.datasets.teeth3ds_processed import create_segmentation_dataloader
from src.models.dgcnn import DGCNNSegmentation


class TestDGCNNForward(unittest.TestCase):
    def test_forward_shape_from_dataloader_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = write_segmentation_fixture(root, samples=2, points=1024)
            with patch.dict("os.environ", {"DATA_DIR": tmpdir}, clear=False):
                loader = create_segmentation_dataloader(config_path, split="train", limit=2)
                batch = next(iter(loader))
        x = batch["x"][:, :1024, :]

        model = DGCNNSegmentation(input_channels=6, num_classes=17, k=20)
        model.eval()

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (batch["x"].shape[0], 1024, 17))
        self.assertEqual(logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
