import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


BatchPreprocessor = Callable[..., Any]


def build_sampling_preprocessors(data_config: Mapping[str, Any], seed: int) -> tuple[BatchPreprocessor | None, BatchPreprocessor | None]:
    sampling_config = data_config.get("sampling", {})
    num_points = sampling_config.get("num_points")
    if num_points is None:
        return None, None

    method = sampling_config.get("method", "random")
    sampler_seed = int(sampling_config.get("seed", seed))

    if method == "random":
        generator = torch.Generator().manual_seed(sampler_seed)

        def preprocess(batch: Any, **_: Any) -> Any:
            return random_sample_point_batch(batch, num_points=int(num_points), generator=generator)

        return preprocess, preprocess

    if method == "overlapping_multiview":
        num_views = int(sampling_config.get("num_views", 4))
        core_points = sampling_config.get("core_points")
        eval_view = int(sampling_config.get("eval_view", 0))
        core_points = int(core_points) if core_points is not None else None

        def train_preprocess(batch: Any, epoch: int = 0, **_: Any) -> Any:
            view_id = max(0, int(epoch) - 1) % max(1, num_views)
            return overlapping_multiview_sample_point_batch(
                batch,
                num_points=int(num_points),
                core_points=core_points,
                view_id=view_id,
                num_views=num_views,
                seed=sampler_seed,
            )

        def eval_preprocess(batch: Any, view_id: int | None = None, **_: Any) -> Any:
            return overlapping_multiview_sample_point_batch(
                batch,
                num_points=int(num_points),
                core_points=core_points,
                view_id=eval_view if view_id is None else int(view_id),
                num_views=num_views,
                seed=sampler_seed,
            )

        return train_preprocess, eval_preprocess

    raise ValueError(f"Unsupported sampling method: {method!r}")


def eval_view_ids_from_config(data_config: Mapping[str, Any]) -> list[int] | None:
    sampling_config = data_config.get("sampling", {})
    if sampling_config.get("method") != "overlapping_multiview":
        return None

    eval_views = sampling_config.get("eval_views")
    if eval_views is None:
        return [int(sampling_config.get("eval_view", 0))]
    if isinstance(eval_views, str):
        if eval_views.lower() != "all":
            raise ValueError("sampling.eval_views must be 'all' or a list of view ids")
        return list(range(int(sampling_config.get("num_views", 1))))
    if isinstance(eval_views, Sequence):
        return [int(view_id) for view_id in eval_views]
    raise TypeError("sampling.eval_views must be 'all' or a list of view ids")


def random_sample_point_batch(
    batch: Any,
    num_points: int | None,
    generator: torch.Generator | None = None,
) -> Any:
    shape = _point_batch_shape(batch)
    if num_points is None or num_points <= 0 or shape is None:
        return batch

    batch_size, num_vertices = shape
    sample_count = int(num_points)
    indices = []
    for _ in range(batch_size):
        if sample_count <= num_vertices:
            indices.append(torch.randperm(num_vertices, generator=generator)[:sample_count])
        else:
            indices.append(torch.randint(num_vertices, (sample_count,), generator=generator))

    return sample_point_batch_by_indices(batch, indices)


def overlapping_multiview_sample_point_batch(
    batch: Any,
    num_points: int | None,
    core_points: int | None = None,
    view_id: int = 0,
    num_views: int = 1,
    seed: int = 0,
) -> Any:
    shape = _point_batch_shape(batch)
    if num_points is None or num_points <= 0 or shape is None:
        return batch

    batch_size, num_vertices = shape
    sample_count = min(int(num_points), num_vertices)
    if sample_count >= num_vertices:
        return batch

    indices = [
        multiview_point_indices(
            num_vertices=num_vertices,
            num_points=sample_count,
            core_points=core_points,
            view_id=view_id,
            num_views=num_views,
            seed=seed,
            scan_id=_scan_id_for_batch_item(batch, batch_index),
        )
        for batch_index in range(batch_size)
    ]
    return sample_point_batch_by_indices(batch, indices)


def multiview_point_indices(
    num_vertices: int,
    num_points: int,
    core_points: int | None = None,
    view_id: int = 0,
    num_views: int = 1,
    seed: int = 0,
    scan_id: str = "",
) -> torch.Tensor:
    sample_count = min(int(num_points), int(num_vertices))
    core_count = int(round(sample_count * 0.75)) if core_points is None else int(core_points)
    core_count = max(0, min(core_count, sample_count))
    extra_count = sample_count - core_count

    generator = torch.Generator().manual_seed(_stable_seed(seed, scan_id))
    order = torch.randperm(num_vertices, generator=generator)
    core = order[:core_count]
    if extra_count <= 0 or core_count >= num_vertices:
        return core.sort()[0]

    remaining = order[core_count:]
    start = (int(view_id) % max(1, int(num_views))) * extra_count
    extra = _circular_slice(remaining, start=start, length=extra_count)
    return torch.cat([core, extra]).sort()[0]


def sample_point_batch_by_indices(batch: Mapping[str, Any], indices: Sequence[torch.Tensor]) -> dict[str, Any]:
    shape = _point_batch_shape(batch)
    if shape is None:
        return dict(batch)

    batch_size, num_vertices = shape
    sampled = dict(batch)
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == batch_size and value.shape[1] == num_vertices:
            sampled[key] = torch.stack(
                [
                    value[batch_index].index_select(0, indices[batch_index].to(value.device))
                    for batch_index in range(batch_size)
                ],
                dim=0,
            )
    return sampled


def _point_batch_shape(batch: Any) -> tuple[int, int] | None:
    if not isinstance(batch, Mapping) or not torch.is_tensor(batch.get("x")):
        return None
    x = batch["x"]
    if x.ndim < 3:
        return None
    return int(x.shape[0]), int(x.shape[1])


def _scan_id_for_batch_item(batch: Mapping[str, Any], batch_index: int) -> str:
    scan_id = batch.get("scan_id")
    if isinstance(scan_id, Sequence) and not isinstance(scan_id, (str, bytes)):
        if batch_index < len(scan_id):
            return str(scan_id[batch_index])
    if scan_id is not None:
        return str(scan_id)
    return f"batch_index:{batch_index}"


def _circular_slice(values: torch.Tensor, start: int, length: int) -> torch.Tensor:
    if values.numel() == 0 or length <= 0:
        return values[:0]

    length = min(int(length), int(values.numel()))
    start = int(start) % int(values.numel())
    end = start + length
    if end <= values.numel():
        return values[start:end]
    return torch.cat([values[start:], values[: end - values.numel()]], dim=0)


def _stable_seed(seed: int, *parts: Any) -> int:
    text = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % (2**63 - 1)
