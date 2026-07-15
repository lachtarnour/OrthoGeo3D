#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.datasets.teeth3ds_raw import (
    build_processed_sample,
    discover_raw_scans,
    load_raw_scan,
)
from src.utils.io import ensure_dir, save_processed_sample
from src.utils.logger import get_logger
from src.utils.paths import (
    get_landmark_dir,
    get_processed_dir,
    get_split_dir,
    get_teeth3ds_dir,
)


SPLITS = ("train", "val")
logger = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess the TeethSeg22 train/validation scans."
    )
    parser.add_argument("--raw_dir", default=str(get_teeth3ds_dir()))
    parser.add_argument("--landmark_dir", default=str(get_landmark_dir()))
    parser.add_argument("--split_source", default="teethseg22")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--out_dir")
    parser.add_argument("--num_points", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    selected_splits = (args.split,) if args.split else SPLITS

    records = discover_raw_scans(args.raw_dir, args.landmark_dir)
    for split in selected_splits:
        preprocess_split(
            records=records,
            split=split,
            num_points=args.num_points,
            seed=args.seed,
            num_workers=args.num_workers,
            skip_existing=args.skip_existing,
            split_source=args.split_source,
            out_root=Path(args.out_dir) if args.out_dir else None,
            limit=args.limit,
        )


def preprocess_split(
    records: list,
    split: str,
    num_points: int,
    seed: int,
    num_workers: int,
    skip_existing: bool,
    split_source: str,
    out_root: Path | None,
    limit: int | None,
) -> None:
    split_file = get_split_dir(split_source) / f"{split}.txt"
    allowed_ids = read_split_file(split_file)
    selected = [record for record in records if record.scan_id in allowed_ids]
    if limit is not None:
        selected = selected[: int(limit)]

    processed_root = out_root or get_processed_dir(split_source)
    out_dir = ensure_dir(processed_root / split)
    worker_args = [
        (offset, record, out_dir, num_points, seed)
        for offset, record in enumerate(selected)
        if not skip_existing or not (out_dir / f"{record.scan_id}.pt").is_file()
    ]
    errors = run_workers(
        worker_args, num_workers=num_workers, description=f"preprocess {split}"
    )

    logger.info("%s: selected %s scans", split, len(selected))
    logger.info(
        "%s: wrote %s new samples to %s", split, len(worker_args) - len(errors), out_dir
    )
    if errors:
        logger.warning("%s: skipped %s scans", split, len(errors))
    write_skip_report(processed_root / "_reports" / f"skipped_{split}.csv", errors)


def run_workers(
    worker_args: list[tuple],
    num_workers: int,
    description: str,
) -> list[tuple[str, str]]:
    errors = []
    if max(1, num_workers) == 1:
        results = (_preprocess_one_record(item) for item in worker_args)
        for result in tqdm(results, total=len(worker_args), desc=description):
            if result[1] is not None:
                errors.append((result[0], result[1]))
                logger.warning("Skipped %s: %s", result[0], result[1])
        return errors

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_preprocess_one_record, item) for item in worker_args
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            scan_id, error = future.result()
            if error is not None:
                errors.append((scan_id, error))
                logger.warning("Skipped %s: %s", scan_id, error)
    return errors


def _preprocess_one_record(args: tuple) -> tuple[str, str | None]:
    offset, record, out_dir, num_points, seed = args
    try:
        sample = build_processed_sample(
            load_raw_scan(record),
            num_points=num_points,
            seed=seed + offset,
            require_labels=True,
        )
        save_processed_sample(sample, out_dir / f"{record.scan_id}.pt")
        return record.scan_id, None
    except Exception as exc:
        return record.scan_id, str(exc)


def read_split_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_skip_report(path: Path, errors: list[tuple[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("scan_id", "error"))
        writer.writeheader()
        for scan_id, error in sorted(errors):
            writer.writerow({"scan_id": scan_id, "error": error})


if __name__ == "__main__":
    main()
