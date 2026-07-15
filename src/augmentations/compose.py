from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from src.augmentations.point_cloud import (
    RandomJitter,
    RandomRotation3D,
    RandomScale,
    RandomTranslation,
)
from src.datasets.teeth3ds_processed import build_features
from src.utils.random import stable_seed


PointCloudTransform = Callable[..., dict[str, Any]]
AUGMENTATION_CONFIG_KEYS = {
    "rotation_degrees",
    "scale_range",
    "translation_range",
    "jitter_std",
    "jitter_clip",
}


class Compose:
    """Compose point-cloud transforms using a shared random generator."""

    def __init__(self, transforms: Sequence[PointCloudTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(
        self,
        point_cloud: Mapping[str, Any],
        *,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        transformed = dict(point_cloud)
        for transform in self.transforms:
            transformed = transform(transformed, generator=generator)
        return transformed


class BatchPointCloudAugmentation:
    """Apply one reproducible random transform per scan and epoch."""

    def __init__(
        self,
        transform: Compose,
        seed: int,
        feature_keys: Sequence[str],
    ) -> None:
        self.transform = transform
        self.seed = int(seed)
        self.feature_keys = tuple(feature_keys)

    def __call__(
        self,
        batch: Mapping[str, Any],
        epoch: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        pos, normal, scan_ids = _batch_geometry(batch)
        positions = []
        normals = []

        for index, scan_id in enumerate(scan_ids):
            generator = torch.Generator(device=pos.device).manual_seed(
                stable_seed(self.seed, "augmentation", scan_id, int(epoch))
            )
            transformed = self.transform(
                {"pos": pos[index], "normal": normal[index]},
                generator=generator,
            )
            positions.append(transformed["pos"])
            normals.append(transformed["normal"])

        augmented = dict(batch)
        augmented["pos"] = torch.stack(positions)
        augmented["normal"] = torch.stack(normals)
        augmented["x"] = build_features(augmented, self.feature_keys)
        return augmented


def build_train_augmentation(
    config: Mapping[str, Any] | None,
    seed: int,
    feature_keys: Sequence[str],
) -> BatchPointCloudAugmentation | None:
    if config is None:
        return None
    _validate_config(config)
    transform = Compose(
        (
            RandomRotation3D(config["rotation_degrees"]),
            RandomScale(config["scale_range"]),
            RandomTranslation(config["translation_range"]),
            RandomJitter(config["jitter_std"], config["jitter_clip"]),
        )
    )
    return BatchPointCloudAugmentation(transform, seed, feature_keys)


def _validate_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - AUGMENTATION_CONFIG_KEYS
    missing = AUGMENTATION_CONFIG_KEYS - set(config)
    if unknown:
        raise ValueError(f"Unsupported augmentation keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing augmentation keys: {sorted(missing)}")


def _batch_geometry(
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Sequence[Any]]:
    pos = batch.get("pos")
    normal = batch.get("normal")
    scan_ids = batch.get("scan_id")
    if not torch.is_tensor(pos) or pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("Batch positions must have shape [B, N, 3]")
    if not torch.is_tensor(normal) or normal.shape != pos.shape:
        raise ValueError("Batch normals must have the same shape as positions")
    if (
        not isinstance(scan_ids, Sequence)
        or isinstance(scan_ids, (str, bytes))
        or len(scan_ids) != pos.shape[0]
    ):
        raise ValueError("Batch must contain one scan_id per point cloud")
    return pos, normal, scan_ids
