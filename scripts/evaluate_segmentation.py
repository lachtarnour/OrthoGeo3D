#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.train_segmentation import (
    build_model,
    configure_cuda_performance,
    validate_model_dataset_contract,
)
from src.datasets.teeth3ds_processed import (
    Teeth3DSSegmentationDataset,
    create_segmentation_dataloader,
)
from src.training.metrics import segmentation_metrics_from_confusion
from src.training.checkpointing import checkpoint_configs
from src.training.sampling import build_sampling_preprocessors, eval_view_ids
from src.training.tasks import SegmentationTask
from src.training.utils import get_device, move_to_device, set_seed
from src.utils.paths import resolve_project_path


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config, data_config = checkpoint_configs(checkpoint)
    seed = int(train_config["run"]["seed"])
    set_seed(seed)

    device = get_device(args.device)
    configure_cuda_performance(train_config.get("performance", {}), device)
    val_loader = create_segmentation_dataloader(
        data_config,
        split="val",
    )
    val_dataset = val_loader.dataset
    if not isinstance(val_dataset, Teeth3DSSegmentationDataset):
        raise TypeError("val_loader.dataset must be a Teeth3DSSegmentationDataset")

    model_config = train_config["model"]
    validate_model_dataset_contract(model_config, val_dataset)

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    task = SegmentationTask(
        num_classes=int(model_config["num_classes"]),
        loss_config=train_config["loss"],
    )
    _, eval_preprocessor = build_sampling_preprocessors(
        train_config["sampling"], seed=seed
    )
    view_ids = eval_view_ids(train_config["sampling"])

    print(
        f"checkpoint={checkpoint_path} epoch={checkpoint.get('epoch')} "
        f"device={device} val_scans={len(val_dataset)} views={len(view_ids)}"
    )
    metrics = evaluate(
        model,
        task,
        val_loader,
        device=device,
        preprocessor=eval_preprocessor,
        view_ids=view_ids,
        amp=bool(train_config["training"]["amp"]),
        progress_every=args.progress_every,
    )
    print_metrics(metrics)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation checkpoint on fixed point views."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    task: SegmentationTask,
    loader,
    device: torch.device,
    preprocessor,
    view_ids: list[int | None],
    amp: bool,
    progress_every: int,
) -> dict[str, Any]:
    model.eval()
    matrix = torch.zeros((task.num_classes, task.num_classes), dtype=torch.long)
    loss_total = 0.0
    point_total = 0
    processed_batches = 0
    started = time.monotonic()
    total_batches = len(loader) * len(view_ids)

    for view_id in view_ids:
        for batch in loader:
            batch = preprocessor(batch, epoch=0, split="val", view_id=view_id)
            batch = move_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(amp and device.type == "cuda"),
            ):
                output = task.validation_step(model, batch)
            matrix += output["confusion_matrix"].detach().cpu()
            num_points = int(output["num_points"])
            loss_total += float(output["loss"].detach().cpu().item()) * num_points
            point_total += num_points
            processed_batches += 1
            del output, batch
            release_device_cache(device, processed_batches)
            if progress_every > 0 and processed_batches % progress_every == 0:
                print(
                    f"eval {processed_batches}/{total_batches} "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )

    metrics: dict[str, Any] = segmentation_metrics_from_confusion(matrix.float())
    metrics["loss"] = loss_total / max(1, point_total)
    metrics["eval_views"] = len(view_ids)
    metrics["scans"] = len(loader.dataset)
    metrics["per_class_iou"] = per_class_iou(matrix)
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics


def per_class_iou(matrix: torch.Tensor) -> list[float | None]:
    matrix = matrix.float()
    true_positive = matrix.diag()
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - true_positive
    return [
        float(true_positive[i] / union[i]) if union[i] > 0 else None
        for i in range(matrix.shape[0])
    ]


def release_device_cache(device: torch.device, batch_index: int) -> None:
    if device.type == "mps" and batch_index % 5 == 0:
        torch.mps.empty_cache()


def print_metrics(metrics: dict[str, Any]) -> None:
    print(
        f"accuracy={metrics['accuracy']:.6f} "
        f"loss={metrics['loss']:.6f} "
        f"mean_f1={metrics['mean_f1']:.6f} "
        f"miou={metrics['miou']:.6f}"
    )


if __name__ == "__main__":
    main()
