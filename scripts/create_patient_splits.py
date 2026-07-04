#!/usr/bin/env python

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.teeth3ds_raw import discover_raw_scans
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.paths import (
    get_landmark_dir,
    get_split_dir,
    get_teeth3ds_dir,
    get_teeth3ds_train_test_split_dir,
)

SPLITS = ("train", "val", "test")
JAW_FILES = {"train": "training", "val": "validation", "test": "testing"}
logger = get_logger("create_patient_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Teeth3DS split files.")
    parser.add_argument("--source", choices=("patient_random", "from_files"), default="patient_random")
    parser.add_argument("--raw_dir", default=str(get_teeth3ds_dir()))
    parser.add_argument("--landmark_dir", default=str(get_landmark_dir()))
    parser.add_argument("--out_dir", help="Default: data/splits/<source>")
    parser.add_argument("--train_files", nargs="*")
    parser.add_argument("--test_files", nargs="*")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_unlabeled", action="store_true")
    args = parser.parse_args()

    records = discover_raw_scans(args.raw_dir, args.landmark_dir)
    if not args.include_unlabeled:
        records = [r for r in records if r.annotation_path is not None]

    if args.source == "patient_random":
        split_records = patient_random_split(records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    else:
        split_dir = get_teeth3ds_train_test_split_dir()
        train_files = args.train_files or [split_dir / "training_lower.txt", split_dir / "training_upper.txt"]
        test_files = args.test_files or [split_dir / "testing_lower.txt", split_dir / "testing_upper.txt"]
        split_records = split_from_files(records, train_files, test_files, args.val_ratio, args.seed)

    out_dir = ensure_dir(args.out_dir or get_split_dir(args.source))
    save_split(out_dir, split_records)

    logger.info("Wrote %s splits to %s", args.source, out_dir)
    for split in SPLITS:
        records = split_records[split]
        logger.info("%s: %s patients, %s scans", split, len({r.patient_id for r in records}), len(records))


def patient_random_split(records, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1.0e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    by_patient = group_by_patient(records)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    n_train = round(len(patients) * train_ratio)
    n_val = round(len(patients) * val_ratio)
    return {
        "train": records_for_patients(by_patient, patients[:n_train]),
        "val": records_for_patients(by_patient, patients[n_train : n_train + n_val]),
        "test": records_for_patients(by_patient, patients[n_train + n_val :]),
    }


def split_from_files(records, train_files, test_files, val_ratio: float, seed: int):
    by_id = {r.scan_id: r for r in records}
    train_records = [by_id[i] for i in read_scan_ids(train_files) if i in by_id]
    test_records = sorted([by_id[i] for i in read_scan_ids(test_files) if i in by_id], key=lambda r: r.scan_id)

    by_patient = group_by_patient(train_records)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    n_val = max(1, round(len(patients) * val_ratio))

    return {
        "train": records_for_patients(by_patient, patients[n_val:]),
        "val": records_for_patients(by_patient, patients[:n_val]),
        "test": test_records,
    }


def group_by_patient(records):
    by_patient = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    return by_patient


def records_for_patients(by_patient, patients):
    return sorted([r for p in patients for r in by_patient[p]], key=lambda r: r.scan_id)


def read_scan_ids(paths) -> list[str]:
    scan_ids = []
    for path in map(Path, paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        scan_ids.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return scan_ids


def save_split(out_dir: Path, split_records) -> None:
    for split in SPLITS:
        records = split_records[split]
        scan_ids = [r.scan_id for r in records]
        patients = sorted({r.patient_id for r in records})
        prefix = JAW_FILES[split]

        (out_dir / f"{split}.txt").write_text("\n".join(scan_ids) + "\n", encoding="utf-8")
        (out_dir / f"{split}_patients.txt").write_text("\n".join(patients) + "\n", encoding="utf-8")
        (out_dir / f"{prefix}_lower.txt").write_text(
            "\n".join(r.scan_id for r in records if r.jaw == "lower") + "\n",
            encoding="utf-8",
        )
        (out_dir / f"{prefix}_upper.txt").write_text(
            "\n".join(r.scan_id for r in records if r.jaw == "upper") + "\n",
            encoding="utf-8",
        )

    with (out_dir / "split_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "patients", "scans", "upper", "lower", "with_annotations", "with_landmarks"])
        for split in SPLITS:
            records = split_records[split]
            writer.writerow(
                [
                    split,
                    len({r.patient_id for r in records}),
                    len(records),
                    sum(r.jaw == "upper" for r in records),
                    sum(r.jaw == "lower" for r in records),
                    sum(r.annotation_path is not None for r in records),
                    sum(r.landmark_path is not None for r in records),
                ]
            )

    patients = {split: {r.patient_id for r in split_records[split]} for split in SPLITS}
    with (out_dir / "patient_overlaps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split_a", "split_b", "overlap_patients"])
        for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test")):
            writer.writerow([split_a, split_b, len(patients[split_a] & patients[split_b])])


if __name__ == "__main__":
    main()
