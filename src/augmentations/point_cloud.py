import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


PointCloud = Mapping[str, Any]


class RandomRotation3D:
    """Apply a small rigid rotation to positions and normals."""

    def __init__(self, max_degrees: Sequence[float]) -> None:
        if len(max_degrees) != 3:
            raise ValueError("max_degrees must contain x, y and z limits")
        self.max_degrees = tuple(float(value) for value in max_degrees)
        if any(value < 0.0 for value in self.max_degrees):
            raise ValueError("Rotation limits must be non-negative")

    def __call__(
        self,
        point_cloud: PointCloud,
        *,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        pos, normal = _geometry(point_cloud)
        limits = pos.new_tensor(self.max_degrees) * (math.pi / 180.0)
        angles = (torch.rand(3, generator=generator, device=pos.device) * 2.0 - 1.0)
        rotation = _rotation_matrix_xyz(angles.to(pos.dtype) * limits)

        transformed = dict(point_cloud)
        transformed["pos"] = pos @ rotation.T
        transformed["normal"] = F.normalize(normal @ rotation.T, dim=-1)
        return transformed


class RandomScale:
    """Apply isotropic scaling around the normalized scan origin."""

    def __init__(self, scale_range: Sequence[float]) -> None:
        if len(scale_range) != 2:
            raise ValueError("scale_range must contain minimum and maximum values")
        self.minimum = float(scale_range[0])
        self.maximum = float(scale_range[1])
        if not 0.0 < self.minimum <= self.maximum:
            raise ValueError("scale_range must satisfy 0 < minimum <= maximum")

    def __call__(
        self,
        point_cloud: PointCloud,
        *,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        pos = _positions(point_cloud)
        scale = pos.new_empty(()).uniform_(
            self.minimum,
            self.maximum,
            generator=generator,
        )
        transformed = dict(point_cloud)
        transformed["pos"] = pos * scale
        return transformed


class RandomTranslation:
    """Translate the complete scan by one random 3D offset."""

    def __init__(self, maximum: float) -> None:
        self.maximum = float(maximum)
        if self.maximum < 0.0:
            raise ValueError("Translation limit must be non-negative")

    def __call__(
        self,
        point_cloud: PointCloud,
        *,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        pos = _positions(point_cloud)
        translation = pos.new_empty(3).uniform_(
            -self.maximum,
            self.maximum,
            generator=generator,
        )
        transformed = dict(point_cloud)
        transformed["pos"] = pos + translation
        return transformed


class RandomJitter:
    """Add clipped Gaussian noise independently to each position."""

    def __init__(self, standard_deviation: float, clip: float) -> None:
        self.standard_deviation = float(standard_deviation)
        self.clip = float(clip)
        if self.standard_deviation < 0.0 or self.clip < 0.0:
            raise ValueError("Jitter standard deviation and clip must be non-negative")

    def __call__(
        self,
        point_cloud: PointCloud,
        *,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        pos = _positions(point_cloud)
        noise = pos.new_empty(pos.shape).normal_(
            mean=0.0,
            std=self.standard_deviation,
            generator=generator,
        )
        transformed = dict(point_cloud)
        transformed["pos"] = pos + noise.clamp(-self.clip, self.clip)
        return transformed


def _positions(point_cloud: PointCloud) -> torch.Tensor:
    pos = point_cloud.get("pos")
    if not torch.is_tensor(pos) or pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError("Point-cloud positions must have shape [N, 3]")
    return pos


def _geometry(point_cloud: PointCloud) -> tuple[torch.Tensor, torch.Tensor]:
    pos = _positions(point_cloud)
    normal = point_cloud.get("normal")
    if not torch.is_tensor(normal) or normal.shape != pos.shape:
        raise ValueError("Point-cloud normals must have the same shape as positions")
    return pos, normal


def _rotation_matrix_xyz(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    return torch.stack(
        (
            torch.stack(
                (cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy)
            ),
            torch.stack(
                (cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx)
            ),
            torch.stack((-sy, cy * sx, cx * cy)),
        )
    )
