#!/usr/bin/env python

import argparse
import subprocess
from pathlib import Path
from zipfile import ZipFile

from src.utils.logger import get_logger
from src.utils.paths import get_download_dir, get_landmark_dir, get_teeth3ds_dir

logger = get_logger("download")

DOWNLOAD_DIR = get_download_dir()
TEETH3DS_DIR = get_teeth3ds_dir()
LANDMARK_DIR = get_landmark_dir()

TEETH3DS_ARCHIVES = {
    "data_part_1.zip": "https://osf.io/download/qhprs/",
    "data_part_2.zip": "https://osf.io/download/4pwnr/",
    "data_part_3.zip": "https://osf.io/download/frwdp/",
    "data_part_4.zip": "https://osf.io/download/2arn4/",
    "data_part_5.zip": "https://osf.io/download/xrz5f/",
    "data_part_6.zip": "https://osf.io/download/23hgq/",
    "data_part_7.zip": "https://osf.io/download/u83ad/",
    "train_test_split.zip": "https://files.de-1.osf.io/v1/resources/xctdy/providers/osfstorage/?zip=",
}

LANDMARK_ARCHIVES = {
    "3DTeethLand_landmarks_train.zip": "https://osf.io/download/k5hbj/",
    "3DTeethLand_landmarks_test.zip": "https://osf.io/download/sqw5e/",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract Teeth3DS and 3DTeethLand annotations."
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download archives without extracting them.",
    )
    parser.add_argument(
        "--force_extract",
        action="store_true",
        help="Extract archives even if target files already exist.",
    )
    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TEETH3DS_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    download_group("Teeth3DS", TEETH3DS_ARCHIVES)
    download_group("3DTeethLand landmarks", LANDMARK_ARCHIVES)

    if not args.download_only:
        extract_group("Teeth3DS", TEETH3DS_ARCHIVES, TEETH3DS_DIR, args.force_extract)
        extract_group(
            "3DTeethLand landmarks", LANDMARK_ARCHIVES, LANDMARK_DIR, args.force_extract
        )

    print_summary()


def download_group(title: str, archives: dict[str, str]) -> None:
    logger.info("== Download %s ==", title)
    for filename, url in archives.items():
        out_path = DOWNLOAD_DIR / filename
        if is_valid_zip(out_path):
            logger.info("OK    %s", filename)
            continue

        logger.info("GET   %s", filename)
        download_file(url, out_path)
        if not is_valid_zip(out_path):
            raise RuntimeError(f"Downloaded file is not a valid zip: {out_path}")


def download_file(url: str, out_path: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--continue-at",
            "-",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "-o",
            str(out_path),
            url,
        ],
        check=True,
    )


def extract_group(
    title: str, archives: dict[str, str], target_dir: Path, force: bool
) -> None:
    logger.info("== Extract %s ==", title)
    for filename in archives:
        zip_path = DOWNLOAD_DIR / filename
        marker = target_dir / ".extract_markers" / f"{filename}.done"
        if marker.exists() and not force:
            logger.info("OK    %s", filename)
            continue

        logger.info("ZIP   %s", filename)
        with ZipFile(zip_path) as archive:
            archive.extractall(target_dir)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n", encoding="utf-8")


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False


def print_summary() -> None:
    logger.info("== Summary ==")
    logger.info("archives : %s", count_files(DOWNLOAD_DIR, "*.zip"))
    logger.info("obj      : %s", count_files(TEETH3DS_DIR, "*.obj"))
    logger.info("json     : %s", count_files(TEETH3DS_DIR, "*.json"))
    logger.info("txt      : %s", count_files(TEETH3DS_DIR, "*.txt"))
    logger.info("landmark : %s", count_files(LANDMARK_DIR, "*__kpt.json"))


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for path in root.rglob(pattern) if path.is_file())


if __name__ == "__main__":
    main()
