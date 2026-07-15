#!/usr/bin/env python

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.datasets.teeth3ds_raw import discover_raw_scans
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.paths import get_landmark_dir, get_split_dir, get_teeth3ds_dir


SPLIT_FILES = {
    "train": ("public-training-set-1.txt", "public-training-set-2.txt"),
    "val": ("private-testing-set.txt",),
}
SPLIT_SOURCES = ("teethseg22", "patient_random")
JAW_FILE_PREFIX = {"train": "training", "val": "validation"}
logger = get_logger("create_patient_splits")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create patient-level train/validation splits."
    )
    parser.add_argument("--source", choices=SPLIT_SOURCES, default="teethseg22")
    parser.add_argument("--raw_dir", default=str(get_teeth3ds_dir()))
    parser.add_argument("--landmark_dir", default=str(get_landmark_dir()))
    parser.add_argument("--out_dir")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [
        record
        for record in discover_raw_scans(args.raw_dir, args.landmark_dir)
        if record.annotation_path is not None
    ]
    if args.source == "teethseg22":
        split_dir = Path(args.raw_dir) / "3DTeethSeg22_challenge_train_test_split"
        split_records = build_teethseg22_split(records, split_dir)
    else:
        split_records = build_patient_random_split(
            records,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    out_dir = ensure_dir(args.out_dir or get_split_dir(args.source))
    save_split(out_dir, split_records)

    logger.info("Wrote %s splits to %s", args.source, out_dir)
    for split, split_items in split_records.items():
        patients = {record.patient_id for record in split_items}
        logger.info("%s: %s patients, %s scans", split, len(patients), len(split_items))


def build_teethseg22_split(records, split_dir: Path) -> dict[str, list]:
    by_id = {record.scan_id: record for record in records}
    split_records = {}
    missing_ids = {}
    for split, filenames in SPLIT_FILES.items():
        scan_ids = read_scan_ids(split_dir / name for name in filenames)
        missing = sorted(scan_id for scan_id in scan_ids if scan_id not in by_id)
        if missing:
            missing_ids[split] = missing
        split_records[split] = sorted(
            (by_id[scan_id] for scan_id in scan_ids if scan_id in by_id),
            key=lambda record: record.scan_id,
        )

    if missing_ids:
        details = ", ".join(
            f"{split}={values[:5]}" for split, values in missing_ids.items()
        )
        raise ValueError(f"TeethSeg22 split references unknown scan ids: {details}")
    return split_records


def build_patient_random_split(
    records,
    train_ratio: float,
    seed: int,
) -> dict[str, list]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    by_patient = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    train_count = round(len(patients) * train_ratio)
    train_patients = set(patients[:train_count])
    return {
        "train": sorted(
            (
                record
                for patient_id in train_patients
                for record in by_patient[patient_id]
            ),
            key=lambda record: record.scan_id,
        ),
        "val": sorted(
            (
                record
                for patient_id in set(patients) - train_patients
                for record in by_patient[patient_id]
            ),
            key=lambda record: record.scan_id,
        ),
    }


def read_scan_ids(paths) -> list[str]:
    scan_ids = []
    for path in map(Path, paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        scan_ids.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return scan_ids


def save_split(out_dir: Path, split_records: dict[str, list]) -> None:
    clear_split_files(out_dir)
    for split, records in split_records.items():
        scan_ids = [record.scan_id for record in records]
        patients = sorted({record.patient_id for record in records})
        prefix = JAW_FILE_PREFIX[split]
        (out_dir / f"{split}.txt").write_text(
            "\n".join(scan_ids) + "\n",
            encoding="utf-8",
        )
        (out_dir / f"{split}_patients.txt").write_text(
            "\n".join(patients) + "\n",
            encoding="utf-8",
        )
        for jaw in ("lower", "upper"):
            jaw_ids = [record.scan_id for record in records if record.jaw == jaw]
            (out_dir / f"{prefix}_{jaw}.txt").write_text(
                "\n".join(jaw_ids) + "\n",
                encoding="utf-8",
            )

    write_split_reports(out_dir, split_records)


def write_split_reports(out_dir: Path, split_records: dict[str, list]) -> None:
    with (out_dir / "split_stats.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["split", "patients", "scans", "upper", "lower", "with_landmarks"]
        )
        for split, records in split_records.items():
            writer.writerow(
                [
                    split,
                    len({record.patient_id for record in records}),
                    len(records),
                    sum(record.jaw == "upper" for record in records),
                    sum(record.jaw == "lower" for record in records),
                    sum(record.landmark_path is not None for record in records),
                ]
            )

    patients = {
        split: {record.patient_id for record in records}
        for split, records in split_records.items()
    }
    overlap = patients["train"] & patients["val"]
    with (out_dir / "patient_overlaps.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["split_a", "split_b", "overlap_patients"])
        writer.writerow(["train", "val", len(overlap)])


def clear_split_files(out_dir: Path) -> None:
    names = ["split_stats.csv", "patient_overlaps.csv"]
    for split, prefix in JAW_FILE_PREFIX.items():
        names.extend(
            [
                f"{split}.txt",
                f"{split}_patients.txt",
                f"{prefix}_lower.txt",
                f"{prefix}_upper.txt",
            ]
        )
    for name in names:
        path = out_dir / name
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
