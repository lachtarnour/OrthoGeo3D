#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.teeth3ds_raw import build_processed_sample, discover_raw_scans, load_raw_scan
from src.utils.io import ensure_dir, save_processed_sample
from src.utils.logger import get_logger
from src.utils.paths import get_landmark_dir, get_processed_dir, get_split_dir, get_teeth3ds_dir

logger = get_logger("prepare_data")
SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw Teeth3DS scans into reusable .pt samples.")
    parser.add_argument("--raw_dir", default=str(get_teeth3ds_dir()))
    parser.add_argument("--landmark_dir", default=str(get_landmark_dir()))
    parser.add_argument("--out_dir", help="Default: data/processed/<split_source>")
    parser.add_argument("--split_file", help="Optional file containing one scan_id per line.")
    parser.add_argument("--split", choices=SPLITS, help="Preprocess one named split.")
    parser.add_argument("--all_splits", action="store_true", help="Preprocess all split files for the selected split source.")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, help="Optional subset to use with --all_splits.")
    parser.add_argument("--split_source", default="teethseg22")
    parser.add_argument("--num_points", type=int, default=60000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow_missing_labels", action="store_true")
    parser.add_argument("--skip_existing", action="store_true", help="Do not rewrite samples that already exist.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    landmark_dir = Path(args.landmark_dir) if args.landmark_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else get_processed_dir(args.split_source)
    num_points = args.num_points
    seed = args.seed if args.seed is not None else 42

    records = discover_raw_scans(raw_dir, landmark_dir)

    if args.all_splits and (args.split or args.split_file):
        raise ValueError("--all_splits cannot be combined with --split or --split_file")
    if args.split and args.split_file:
        raise ValueError("--split cannot be combined with --split_file")
    if args.splits and not args.all_splits:
        raise ValueError("--splits can only be used with --all_splits")

    if args.all_splits:
        for split in selected_split_names(args.split_source, args.splits):
            preprocess_split(args, records, split, num_points, seed)
        return

    if args.split:
        preprocess_split(args, records, args.split, num_points, seed)
        return

    split_file = Path(args.split_file) if args.split_file else None
    preprocess_records(
        records=records,
        out_dir=out_dir,
        split_file=split_file,
        num_points=num_points,
        seed=seed,
        num_workers=args.num_workers,
        limit=args.limit,
        require_labels=not args.allow_missing_labels,
        skip_existing=args.skip_existing,
        skip_report_path=out_dir / "_reports" / "skipped.csv",
        desc="preprocess",
    )


def selected_split_names(split_source: str, requested_splits: list[str] | None = None) -> tuple[str, ...]:
    if requested_splits:
        return tuple(requested_splits)
    split_dir = get_split_dir(split_source)
    splits = tuple(split for split in SPLITS if (split_dir / f"{split}.txt").is_file())
    if not splits:
        raise FileNotFoundError(f"No split files found in {split_dir}")
    return splits


def preprocess_split(args: argparse.Namespace, records: list, split: str, num_points: int, seed: int) -> None:
    split_file = get_split_dir(args.split_source) / f"{split}.txt"
    root_out_dir = Path(args.out_dir) if args.out_dir else get_processed_dir(args.split_source)
    out_dir = root_out_dir / split
    preprocess_records(
        records=records,
        out_dir=out_dir,
        split_file=split_file,
        num_points=num_points,
        seed=seed,
        num_workers=args.num_workers,
        limit=args.limit,
        require_labels=not args.allow_missing_labels,
        skip_existing=args.skip_existing,
        skip_report_path=root_out_dir / "_reports" / f"skipped_{split}.csv",
        desc=f"preprocess {split}",
    )


def preprocess_records(
    records: list,
    out_dir: Path,
    split_file: Path | None,
    num_points: int,
    seed: int,
    num_workers: int,
    limit: int | None,
    require_labels: bool,
    skip_existing: bool,
    skip_report_path: Path | None,
    desc: str,
) -> None:
    out_dir = ensure_dir(out_dir)
    allowed_ids = _read_split_file(str(split_file) if split_file else None)
    selected_records = records
    if allowed_ids is not None:
        selected_records = [record for record in selected_records if record.scan_id in allowed_ids]
    if limit:
        selected_records = selected_records[:limit]

    logger.info("Using %s worker(s)", max(1, num_workers))
    errors = []
    all_worker_args = [
        (
            offset,
            record,
            out_dir,
            num_points,
            seed,
            require_labels,
        )
        for offset, record in enumerate(selected_records)
    ]
    worker_args = all_worker_args
    if skip_existing:
        worker_args = [item for item in all_worker_args if not (out_dir / f"{item[1].scan_id}.pt").is_file()]
        skipped_existing = len(all_worker_args) - len(worker_args)
        logger.info("Found %s existing sample(s) in %s", skipped_existing, out_dir)

    if max(1, num_workers) == 1:
        for item in tqdm(worker_args, desc=desc):
            result = _preprocess_one_record(item)
            if result[1] is not None:
                errors.append(result)
                logger.warning("Skipped %s: %s", result[0], result[1])
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_preprocess_one_record, item) for item in worker_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                result = future.result()
                if result[1] is not None:
                    errors.append(result)
                    logger.warning("Skipped %s: %s", result[0], result[1])

    logger.info("Discovered %s scans", len(selected_records))
    logger.info("Wrote %s new samples to %s", len(worker_args) - len(errors), out_dir)
    if errors:
        logger.warning("Skipped %s scans with errors", len(errors))
    if skip_report_path is not None:
        write_skip_report(skip_report_path, errors)


def _preprocess_one_record(args: tuple) -> tuple[str, str | None]:
    (
        offset,
        record,
        out_dir,
        num_points,
        seed,
        require_labels,
    ) = args
    try:
        raw = load_raw_scan(record)
        sample = build_processed_sample(
            raw,
            num_points=num_points,
            seed=seed + offset,
            require_labels=require_labels,
        )
        out_path = out_dir / f"{record.scan_id}.pt"
        save_processed_sample(sample, out_path)
        return record.scan_id, None
    except Exception as exc:
        return record.scan_id, str(exc)


def write_skip_report(path: Path, errors: list[tuple[str, str | None]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("scan_id", "error"))
        writer.writeheader()
        for scan_id, error in sorted(errors, key=lambda item: item[0]):
            writer.writerow({"scan_id": scan_id, "error": error or ""})
    logger.info("Wrote skip report to %s", path)


def _read_split_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


if __name__ == "__main__":
    main()
