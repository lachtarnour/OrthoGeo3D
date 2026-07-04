#!/usr/bin/env python

import argparse
import sys
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
from src.utils.paths import get_landmark_dir, get_processed_dir, get_teeth3ds_dir

logger = get_logger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw Teeth3DS scans into reusable .pt samples.")
    parser.add_argument("--raw_dir", default=str(get_teeth3ds_dir()))
    parser.add_argument("--landmark_dir", default=str(get_landmark_dir()))
    parser.add_argument("--out_dir", default=str(get_processed_dir()))
    parser.add_argument("--split_file", help="Optional file containing one scan_id per line.")
    parser.add_argument("--num_points", type=int, default=30000)
    parser.add_argument("--sampling", default="random")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--curvature", action="store_true", help="Compute discrete mean-curvature features.")
    parser.add_argument("--allow_missing_labels", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    landmark_dir = Path(args.landmark_dir) if args.landmark_dir else None
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    num_points = args.num_points
    seed = args.seed if args.seed is not None else 42

    records = discover_raw_scans(raw_dir, landmark_dir)
    allowed_ids = _read_split_file(args.split_file)
    if allowed_ids is not None:
        records = [record for record in records if record.scan_id in allowed_ids]
    if args.limit:
        records = records[: args.limit]

    errors = []
    for offset, record in enumerate(tqdm(records, desc="preprocess")):
        try:
            raw = load_raw_scan(record)
            sample = build_processed_sample(
                raw,
                num_points=num_points,
                sampling=args.sampling,
                seed=seed + offset,
                require_labels=not args.allow_missing_labels,
                compute_curvature=args.curvature,
            )
            out_path = out_dir / f"{record.scan_id}.pt"
            save_processed_sample(sample, out_path)
        except Exception as exc:
            message = str(exc)
            errors.append((record.scan_id, message))
            logger.warning("Skipped %s: %s", record.scan_id, message)

    logger.info("Discovered %s scans", len(records))
    logger.info("Wrote %s samples to %s", len(records) - len(errors), out_dir)
    if errors:
        logger.warning("Skipped %s scans with errors", len(errors))


def _read_split_file(path: str | None) -> set[str] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


if __name__ == "__main__":
    main()
