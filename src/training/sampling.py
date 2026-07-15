from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from src.utils.random import stable_seed


BatchPreprocessor = Callable[..., dict[str, Any]]
SAMPLING_CONFIG_KEYS = {"source_points", "num_points", "eval_views"}


def build_sampling_preprocessors(
    sampling_config: Mapping[str, Any],
    seed: int,
    train_transform: BatchPreprocessor | None = None,
) -> tuple[BatchPreprocessor, BatchPreprocessor]:
    validate_sampling_config(sampling_config)
    source_points = int(sampling_config["source_points"])
    num_points = int(sampling_config["num_points"])
    eval_views = int(sampling_config["eval_views"])
    if not 0 < num_points <= source_points:
        raise ValueError("sampling.num_points must be in (0, source_points]")
    if eval_views <= 0:
        raise ValueError("sampling.eval_views must be positive")

    def train_preprocess(batch: Any, epoch: int = 1, **_: Any) -> dict[str, Any]:
        sampled = sample_batch(
            batch,
            num_points=num_points,
            source_points=source_points,
            seed=seed,
            namespace="train",
            sample_id=max(1, int(epoch)),
        )
        if train_transform is None:
            return sampled
        return train_transform(sampled, epoch=epoch)

    def eval_preprocess(
        batch: Any,
        view_id: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        selected_view = 0 if view_id is None else int(view_id)
        if not 0 <= selected_view < eval_views:
            raise ValueError(
                f"view_id must be in [0, {eval_views}), got {selected_view}"
            )
        return sample_batch(
            batch,
            num_points=num_points,
            source_points=source_points,
            seed=seed,
            namespace="eval",
            sample_id=selected_view,
        )

    return train_preprocess, eval_preprocess


def eval_view_ids(sampling_config: Mapping[str, Any]) -> list[int]:
    validate_sampling_config(sampling_config)
    return list(range(int(sampling_config["eval_views"])))


def validate_sampling_config(sampling_config: Mapping[str, Any]) -> None:
    unknown = set(sampling_config) - SAMPLING_CONFIG_KEYS
    missing = {"source_points", "num_points", "eval_views"} - set(sampling_config)
    if unknown:
        raise ValueError(f"Unsupported sampling keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing sampling keys: {sorted(missing)}")


def sample_batch(
    batch: Mapping[str, Any],
    num_points: int,
    source_points: int,
    seed: int,
    namespace: str,
    sample_id: int,
) -> dict[str, Any]:
    batch_size, num_vertices = point_batch_shape(batch)
    if num_vertices != source_points:
        raise ValueError(
            f"Sampling expects {source_points} processed points per scan, "
            f"got {num_vertices}"
        )
    indices = [
        deterministic_random_point_indices(
            num_vertices=num_vertices,
            num_points=num_points,
            seed=seed,
            namespace=namespace,
            sample_id=sample_id,
            scan_id=scan_id_for_batch_item(batch, batch_index),
        )
        for batch_index in range(batch_size)
    ]
    return sample_point_batch_by_indices(batch, indices)


def deterministic_random_point_indices(
    num_vertices: int,
    num_points: int,
    seed: int,
    namespace: str,
    sample_id: int,
    scan_id: str,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(
        stable_seed(seed, namespace, scan_id, sample_id)
    )
    return torch.randperm(num_vertices, generator=generator)[:num_points].sort()[0]


def sample_point_batch_by_indices(
    batch: Mapping[str, Any],
    indices: Sequence[torch.Tensor],
) -> dict[str, Any]:
    batch_size, num_vertices = point_batch_shape(batch)
    sampled = dict(batch)
    for key, value in batch.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[:2] == (batch_size, num_vertices)
        ):
            sampled[key] = torch.stack(
                [
                    value[batch_index].index_select(
                        0,
                        indices[batch_index].to(value.device),
                    )
                    for batch_index in range(batch_size)
                ]
            )
    return sampled


def point_batch_shape(batch: Mapping[str, Any]) -> tuple[int, int]:
    x = batch.get("x")
    if not torch.is_tensor(x) or x.ndim != 3:
        raise TypeError("Point sampling expects batch['x'] with shape [B, N, C]")
    return int(x.shape[0]), int(x.shape[1])


def scan_id_for_batch_item(batch: Mapping[str, Any], batch_index: int) -> str:
    scan_ids = batch.get("scan_id")
    if isinstance(scan_ids, Sequence) and not isinstance(scan_ids, (str, bytes)):
        return str(scan_ids[batch_index])
    raise TypeError("Point sampling expects one scan_id per batch item")
