import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.create_patient_splits import predefined_teethseg22_split, save_split, teethseg22_split


class TestCreatePatientSplits(unittest.TestCase):
    def test_teethseg22_split_combines_public_sets_for_train(self) -> None:
        records = [
            SimpleNamespace(scan_id="a_lower"),
            SimpleNamespace(scan_id="a_upper"),
            SimpleNamespace(scan_id="b_lower"),
            SimpleNamespace(scan_id="b_upper"),
            SimpleNamespace(scan_id="c_lower"),
            SimpleNamespace(scan_id="c_upper"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir)
            (split_dir / "public-training-set-1.txt").write_text("a_upper\na_lower\n", encoding="utf-8")
            (split_dir / "public-training-set-2.txt").write_text("b_upper\nb_lower\n", encoding="utf-8")
            (split_dir / "private-testing-set.txt").write_text("c_upper\nc_lower\n", encoding="utf-8")

            split_records = teethseg22_split(records, split_dir)

        self.assertEqual([r.scan_id for r in split_records["train"]], ["a_lower", "a_upper", "b_lower", "b_upper"])
        self.assertEqual([r.scan_id for r in split_records["val"]], ["c_lower", "c_upper"])
        self.assertNotIn("test", split_records)

    def test_teethseg22_3way_keeps_each_official_file_separate(self) -> None:
        records = [
            SimpleNamespace(scan_id="a_lower"),
            SimpleNamespace(scan_id="b_lower"),
            SimpleNamespace(scan_id="c_lower"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir)
            (split_dir / "public-training-set-1.txt").write_text("a_lower\n", encoding="utf-8")
            (split_dir / "public-training-set-2.txt").write_text("b_lower\n", encoding="utf-8")
            (split_dir / "private-testing-set.txt").write_text("c_lower\n", encoding="utf-8")

            split_records = predefined_teethseg22_split(records, split_dir, source="teethseg22_3way")

        self.assertEqual([r.scan_id for r in split_records["train"]], ["a_lower"])
        self.assertEqual([r.scan_id for r in split_records["val"]], ["b_lower"])
        self.assertEqual([r.scan_id for r in split_records["test"]], ["c_lower"])

    def test_save_split_removes_stale_unused_test_files(self) -> None:
        record = SimpleNamespace(
            scan_id="a_lower",
            patient_id="a",
            jaw="lower",
            annotation_path=Path("a.json"),
            landmark_path=Path("a.json"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "test.txt").write_text("stale\n", encoding="utf-8")

            save_split(out_dir, {"train": [record], "val": [record]})

            self.assertTrue((out_dir / "train.txt").is_file())
            self.assertTrue((out_dir / "val.txt").is_file())
            self.assertFalse((out_dir / "test.txt").exists())


if __name__ == "__main__":
    unittest.main()
