import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    value = os.environ.get("DATA_DIR") or _read_env_value(PROJECT_ROOT / ".env", "DATA_DIR")
    if value:
        path = Path(_clean_env_value(value)).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "data"


def get_raw_dir() -> Path:
    return get_data_dir() / "raw"


def get_teeth3ds_dir() -> Path:
    return get_raw_dir() / "Teeth3DS"


def get_teeth3ds_train_test_split_dir() -> Path:
    return get_teeth3ds_dir() / "Teeth3DS_train_test_split"


def get_landmark_dir() -> Path:
    return get_raw_dir() / "Teeth3DSLandmarks"


def get_download_dir() -> Path:
    return get_raw_dir() / "Teeth3DS_downloads"


def get_processed_dir() -> Path:
    return get_data_dir() / "processed"


def get_splits_dir() -> Path:
    return get_data_dir() / "splits"


def _read_env_value(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        return _clean_env_value(value)
    return None


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value
