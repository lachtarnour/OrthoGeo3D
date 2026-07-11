import tempfile
import unittest
from pathlib import Path

from scripts.check_dataset_integrity import load_skip_report, selected_split_names, split_file_errors


class TestDatasetIntegrity(unittest.TestCase):
    def test_selected_split_names_discovers_existing_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.txt").write_text("scan_a\n", encoding="utf-8")
            (root / "val.txt").write_text("scan_b\n", encoding="utf-8")

            self.assertEqual(selected_split_names(root), ("train", "val"))

    def test_documented_missing_scan_is_warning_when_skips_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_file = root / "train.txt"
            split_file.write_text("scan_a\nscan_b\n", encoding="utf-8")
            skip_dir = root / "_reports"
            skip_dir.mkdir()
            (skip_dir / "skipped_train.csv").write_text(
                "scan_id,error\nscan_b,bad annotation\n",
                encoding="utf-8",
            )

            skipped = load_skip_report(str(skip_dir), root, "train")
            errors, warnings = split_file_errors(
                "train",
                split_file,
                [root / "scan_a.pt"],
                skipped=skipped,
                allow_skipped=True,
            )

            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("documented as skipped", warnings[0])

    def test_undocumented_missing_scan_remains_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_file = root / "train.txt"
            split_file.write_text("scan_a\nscan_b\n", encoding="utf-8")

            errors, warnings = split_file_errors(
                "train",
                split_file,
                [root / "scan_a.pt"],
                skipped={},
                allow_skipped=True,
            )

            self.assertEqual(warnings, [])
            self.assertEqual(len(errors), 1)
            self.assertIn("undocumented", errors[0])


if __name__ == "__main__":
    unittest.main()
