import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.datasets.teeth3ds_processed import Teeth3DSSegmentationDataset
from src.utils.paths import get_processed_dir


class TestProcessedPaths(unittest.TestCase):
    def test_processed_dir_is_namespaced_by_split_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"DATA_DIR": tmpdir}, clear=False):
                self.assertEqual(get_processed_dir("teethseg22"), Path(tmpdir) / "processed" / "teethseg22")
                self.assertEqual(get_processed_dir("patient_random"), Path(tmpdir) / "processed" / "patient_random")

    def test_dataset_uses_split_source_when_processed_dir_is_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed_dir = root / "processed" / "teethseg22" / "train"
            processed_dir.mkdir(parents=True)
            (processed_dir / "scan_a.pt").touch()
            config_path = root / "data.yaml"
            config_path.write_text(
                """
paths:
  processed_dir: null
dataset:
  split_source: teethseg22
segmentation_dataset:
  feature_keys: [pos, normal]
  target_key: y_arch_class
""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"DATA_DIR": tmpdir}, clear=False):
                dataset = Teeth3DSSegmentationDataset.from_config(config_path, split="train")

            self.assertEqual(dataset.processed_dir, root / "processed" / "teethseg22")
            self.assertEqual([path.name for path in dataset.paths], ["scan_a.pt"])

    def test_explicit_processed_dir_overrides_split_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            custom_dir = root / "custom_processed"
            (custom_dir / "train").mkdir(parents=True)
            (custom_dir / "train" / "scan_a.pt").touch()
            config_path = root / "data.yaml"
            config_path.write_text(
                f"""
paths:
  processed_dir: {custom_dir}
dataset:
  split_source: teethseg22
segmentation_dataset:
  feature_keys: [pos, normal]
  target_key: y_arch_class
""",
                encoding="utf-8",
            )

            dataset = Teeth3DSSegmentationDataset.from_config(config_path, split="train")

            self.assertEqual(dataset.processed_dir, custom_dir)
            self.assertEqual([path.name for path in dataset.paths], ["scan_a.pt"])


if __name__ == "__main__":
    unittest.main()
