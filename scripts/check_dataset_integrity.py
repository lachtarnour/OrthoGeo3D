#!/usr/bin/env python

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.datasets.labels import ARCH_CLASS_LABELS, FDI_LABELS, map_fdi_to_arch_class
from src.utils.io import load_processed_sample
from src.utils.paths import get_processed_dir, get_split_dir


SPLITS = ("train", "val", "test")
POINT_KEYS = ("pos_raw", "pos", "normal")
LABEL_KEYS = ("y_binary", "y_fdi", "y_fdi_class", "y_instance", "source_indices")
OPTIONAL_LABEL_KEYS = ("y_arch_class",)
REQUIRED_KEYS = {
    "scan_id", "patient_id", "jaw", "center", "scale", "fdi_to_class", "class_to_fdi",
    "landmarks_raw", "landmarks_norm", "landmark_to_tooth", "tooth_centers_raw", "tooth_centers_norm",
    *POINT_KEYS, *LABEL_KEYS,
}


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir) if args.processed_dir else get_processed_dir(args.split_source)
    split_dir = get_split_dir(args.split_source)
    splits = selected_split_names(split_dir, args.splits)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    patients: dict[str, set[str]] = {split: set() for split in splits}
    total_files = total_landmarks = 0

    print("Checking processed dataset")
    print(f"processed_dir: {processed_dir}")
    print(f"split_dir:     {split_dir}\n")

    for split in splits:
        files = sorted((processed_dir / split).glob("*.pt"))
        files = files[: args.limit] if args.limit else files
        if not files:
            all_errors.append(f"{split}: no .pt files found")
            continue
        if not args.limit:
            skipped = load_skip_report(args.skip_report_dir, processed_dir, split) if args.allow_skipped else {}
            errors, warnings = split_file_errors(
                split,
                split_dir / f"{split}.txt",
                files,
                skipped=skipped,
                allow_skipped=args.allow_skipped,
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)

        split_landmarks = 0
        for path in files:
            errors, warnings, has_landmarks, patient_id = check_file(path, expected_num_points=args.expected_num_points)
            all_errors.extend(errors)
            all_warnings.extend(warnings)
            split_landmarks += int(has_landmarks)
            if patient_id:
                patients[split].add(patient_id)

        total_files += len(files)
        total_landmarks += split_landmarks
        print(f"{split:5s}: {len(files):4d} files | {len(patients[split]):4d} patients | {split_landmarks:3d} with landmarks")

    all_errors.extend(patient_overlap_errors(patients))
    print_result(total_files, total_landmarks, all_warnings, all_errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple conformity check for processed OrthoTwin3D .pt files.")
    parser.add_argument("--processed_dir", help="Default: data/processed/<split_source>")
    parser.add_argument("--split_source", default="teethseg22")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, help="Default: discover split files in data/splits/<split_source>")
    parser.add_argument("--limit", type=int, help="Check only the first N files per split.")
    parser.add_argument("--expected_num_points", type=int, help="Require every processed sample to contain this many points.")
    parser.add_argument(
        "--allow_skipped",
        action="store_true",
        help="Allow missing split entries only when they are documented in processed/_reports/skipped_<split>.csv.",
    )
    parser.add_argument("--skip_report_dir", help="Default: <processed_dir>/_reports")
    return parser.parse_args()


def selected_split_names(split_dir: Path, requested_splits: list[str] | None = None) -> tuple[str, ...]:
    if requested_splits:
        return tuple(requested_splits)
    splits = tuple(split for split in SPLITS if (split_dir / f"{split}.txt").is_file())
    if not splits:
        raise FileNotFoundError(f"No split files found in {split_dir}")
    return splits


def split_file_errors(
    split: str,
    split_file: Path,
    files: list[Path],
    skipped: dict[str, str] | None = None,
    allow_skipped: bool = False,
) -> tuple[list[str], list[str]]:
    if not split_file.is_file():
        return [f"{split}: split file not found: {split_file}"], []

    expected_ids = {
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    processed_ids = {path.stem for path in files}
    skipped = skipped or {}
    skipped_ids = set(skipped)

    errors = []
    warnings = []
    missing = sorted(expected_ids - processed_ids)
    unexpected = sorted(processed_ids - expected_ids)
    documented_missing = sorted(set(missing) & skipped_ids)
    undocumented_missing = sorted(set(missing) - skipped_ids)
    stale_skips = sorted(skipped_ids - expected_ids)
    processed_skips = sorted(skipped_ids & processed_ids)

    if missing:
        if allow_skipped:
            if documented_missing:
                warnings.append(
                    f"{split}: {len(documented_missing)} missing processed file(s) documented as skipped, "
                    f"first examples: {documented_missing[:10]}"
                )
            if undocumented_missing:
                errors.append(
                    f"{split}: missing {len(undocumented_missing)} undocumented processed file(s), "
                    f"first examples: {undocumented_missing[:10]}"
                )
        else:
            errors.append(f"{split}: missing {len(missing)} processed file(s), first examples: {missing[:10]}")
    if unexpected:
        errors.append(f"{split}: {len(unexpected)} unexpected processed file(s), first examples: {unexpected[:10]}")
    if stale_skips:
        errors.append(f"{split}: skip report contains {len(stale_skips)} scan_id(s) outside split, first examples: {stale_skips[:10]}")
    if processed_skips:
        warnings.append(f"{split}: skip report contains {len(processed_skips)} already processed scan_id(s), first examples: {processed_skips[:10]}")
    return errors, warnings


def load_skip_report(skip_report_dir: str | None, processed_dir: Path, split: str) -> dict[str, str]:
    root = Path(skip_report_dir) if skip_report_dir else processed_dir / "_reports"
    path = root / f"skipped_{split}.csv"
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row["scan_id"]: row.get("error", "")
            for row in csv.DictReader(f)
            if row.get("scan_id")
        }


def check_file(path: Path, expected_num_points: int | None = None) -> tuple[list[str], list[str], bool, str | None]:
    try:
        sample = load_processed_sample(path)
    except Exception as exc:
        return [f"{path.name}: cannot load ({exc})"], [], False, None

    errors, warnings = check_sample(sample, path.name, expected_num_points=expected_num_points)
    return errors, warnings, bool(sample.get("landmarks_raw")), sample.get("patient_id")


def check_sample(sample: dict, name: str, expected_num_points: int | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED_KEYS - set(sample)
    if missing:
        return [f"{name}: missing keys {sorted(missing)}"], warnings

    n = check_shapes(sample, name, errors)
    if n is None:
        return errors, warnings
    if expected_num_points is not None and n != expected_num_points:
        errors.append(f"{name}: has {n} points, expected {expected_num_points}")

    check_finite(sample, name, errors)
    check_normals(sample["normal"], sample["y_fdi"], name, warnings)
    check_labels(sample, name, errors)
    check_metadata(sample, name, warnings, errors)
    return errors, warnings


def check_shapes(sample: dict, name: str, errors: list[str]) -> int | None:
    pos = sample["pos"]
    if tuple(pos.shape)[-1:] != (3,) or len(pos.shape) != 2:
        errors.append(f"{name}: pos must have shape [N, 3]")
        return None

    n = int(pos.shape[0])
    for key in POINT_KEYS:
        if tuple(sample[key].shape) != (n, 3):
            errors.append(f"{name}: {key} shape is {tuple(sample[key].shape)}, expected ({n}, 3)")
    for key in LABEL_KEYS:
        if tuple(sample[key].shape) != (n,):
            errors.append(f"{name}: {key} shape is {tuple(sample[key].shape)}, expected ({n},)")
    for key in OPTIONAL_LABEL_KEYS:
        if key in sample and tuple(sample[key].shape) != (n,):
            errors.append(f"{name}: {key} shape is {tuple(sample[key].shape)}, expected ({n},)")
    return n


def check_finite(sample: dict, name: str, errors: list[str]) -> None:
    for key in POINT_KEYS:
        if not torch.isfinite(sample[key]).all():
            errors.append(f"{name}: {key} contains NaN or inf")


def check_normals(normal: torch.Tensor, y_fdi: torch.Tensor, name: str, warnings: list[str]) -> None:
    zero_mask = torch.linalg.norm(normal.float(), dim=1) < 1e-6
    zero_on_tooth = int((zero_mask & (y_fdi > 0)).sum().item())
    zero_on_background = int((zero_mask & (y_fdi == 0)).sum().item())
    # Teeth3DS has a few degenerate background/gingiva vertices with no valid normal; they are not used as tooth geometry.
    if zero_on_tooth:
        warnings.append(f"{name}: {zero_on_tooth} zero normal vector(s) on tooth points")


def check_labels(sample: dict, name: str, errors: list[str]) -> None:
    y_fdi = sample["y_fdi"]
    y_class = sample["y_fdi_class"]
    valid_fdi = torch.tensor(FDI_LABELS, dtype=y_fdi.dtype)
    if not torch.isin(y_fdi, valid_fdi).all():
        errors.append(f"{name}: invalid FDI labels {sorted(set(y_fdi.tolist()) - set(FDI_LABELS))}")

    fdi_to_class = {int(k): int(v) for k, v in sample["fdi_to_class"].items()}
    class_to_fdi = {int(k): int(v) for k, v in sample["class_to_fdi"].items()}
    expected_classes = set(range(len(FDI_LABELS)))
    if set(fdi_to_class) != set(FDI_LABELS) or set(fdi_to_class.values()) != expected_classes:
        errors.append(f"{name}: invalid fdi_to_class")
    if set(class_to_fdi) != expected_classes or set(class_to_fdi.values()) != set(FDI_LABELS):
        errors.append(f"{name}: invalid class_to_fdi")

    expected_y_class = torch.empty_like(y_fdi)
    for fdi, class_id in fdi_to_class.items():
        expected_y_class[y_fdi == fdi] = class_id
    if not torch.equal(y_class, expected_y_class):
        errors.append(f"{name}: y_fdi_class is inconsistent with y_fdi")
    if "y_arch_class" in sample:
        y_arch_class = sample["y_arch_class"]
        expected_y_arch_class = torch.as_tensor(map_fdi_to_arch_class(y_fdi.cpu().numpy()), dtype=y_arch_class.dtype)
        if not torch.equal(y_arch_class, expected_y_arch_class):
            errors.append(f"{name}: y_arch_class is inconsistent with y_fdi")
        if int(y_arch_class.min().item()) < 0 or int(y_arch_class.max().item()) >= len(ARCH_CLASS_LABELS):
            errors.append(f"{name}: y_arch_class contains labels outside [0, {len(ARCH_CLASS_LABELS) - 1}]")
    if not torch.equal(sample["y_binary"], (y_fdi > 0).long()):
        errors.append(f"{name}: y_binary is inconsistent with y_fdi")


def check_metadata(sample: dict, name: str, warnings: list[str], errors: list[str]) -> None:
    if sample["jaw"] not in {"upper", "lower"}:
        warnings.append(f"{name}: jaw is {sample['jaw']!r}")
    if not isinstance(sample["scale"], (float, int)) or float(sample["scale"]) <= 0:
        errors.append(f"{name}: scale must be positive")


def patient_overlap_errors(patients: dict[str, set[str]]) -> list[str]:
    errors = []
    splits = list(patients)
    pairs = [(splits[i], splits[j]) for i in range(len(splits)) for j in range(i + 1, len(splits))]
    for left, right in pairs:
        overlap = patients[left] & patients[right]
        if overlap:
            errors.append(f"patient overlap {left}/{right}: {sorted(overlap)[:10]}")
    return errors


def print_result(total_files: int, total_landmarks: int, warnings: list[str], errors: list[str]) -> None:
    print(f"\nTotal checked files: {total_files}")
    print(f"Files with landmarks: {total_landmarks}")
    print(f"Warnings: {len(warnings)}")
    print(f"Errors:   {len(errors)}")
    print_items("Warnings", warnings, limit=20)
    print_items("Errors", errors, limit=30)
    if errors:
        raise SystemExit(1)
    print("\nStatus: OK")


def print_items(title: str, items: list[str], limit: int) -> None:
    if not items:
        return
    print(f"\n{title}:")
    for item in items[:limit]:
        print(f"- {item}")
    if len(items) > limit:
        print(f"- ... {len(items) - limit} more")


if __name__ == "__main__":
    main()
