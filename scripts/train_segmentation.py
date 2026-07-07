#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.teeth3ds_processed import (
    Teeth3DSSegmentationDataset,
    compute_class_counts,
    compute_class_weights,
    create_segmentation_dataloader,
)
from src.models import DGCNNSegmentation
from src.training import CheckpointManager, SegmentationTask, Trainer
from src.training.loggers import CompositeLogger, ConsoleLogger, JsonlLogger, WandbLogger
from src.training.utils import ensure_dir, random_sample_point_batch, set_seed
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_project_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    optim_config = config.get("optim", {})
    logging_config = config.get("logging", {})
    checkpoint_config = config.get("checkpoint", {})

    seed = int(experiment_config.get("seed", 42))
    set_seed(seed)

    run_dir = build_run_dir(experiment_config)
    save_config_copy(config, run_dir / "config.yaml")

    train_loader = create_segmentation_dataloader(
        data_config.get("config_path", "configs/data.yaml"),
        split=data_config.get("train_split", "train"),
        limit=data_config.get("train_limit"),
    )
    val_loader = create_segmentation_dataloader(
        data_config.get("config_path", "configs/data.yaml"),
        split=data_config.get("val_split", "val"),
        limit=data_config.get("val_limit"),
    )

    model = build_model(config.get("model", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_config.get("lr", 1.0e-3)),
        weight_decay=float(optim_config.get("weight_decay", 1.0e-4)),
    )

    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, Teeth3DSSegmentationDataset):
        raise TypeError("train_loader.dataset must be a Teeth3DSSegmentationDataset")
    class_counts = compute_class_counts(train_dataset, num_classes=int(config["model"].get("num_classes", 33)))
    class_weights = compute_class_weights(class_counts)

    task = SegmentationTask(
        num_classes=int(config["model"].get("num_classes", 33)),
        class_weights=class_weights,
    )
    logger = build_logger(logging_config, experiment_config, run_dir, config)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_dir / "checkpoints",
        monitor=checkpoint_config.get("monitor", "val_loss"),
        mode=checkpoint_config.get("mode", "min"),
        save_every_epochs=int(checkpoint_config.get("save_every_epochs", 1)),
    )
    batch_preprocessor = build_batch_preprocessor(data_config, seed=seed)

    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=None,
        device=optim_config.get("device", "auto"),
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        max_epochs=int(optim_config.get("epochs", 100)),
        grad_clip=optim_config.get("grad_clip"),
        amp=bool(optim_config.get("amp", False)),
        batch_preprocessor=batch_preprocessor,
        config=config,
    )

    resume_path = args.resume or checkpoint_config.get("resume")
    if resume_path:
        trainer.resume_from_checkpoint(str(resolve_project_path(resume_path)))

    trainer.fit(train_loader, val_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scan-level FDI segmentation model.")
    parser.add_argument("--config", default="configs/train/dgcnn_segmentation.yaml")
    parser.add_argument("--resume", help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    name = model_config.get("name", "dgcnn_segmentation")
    if name != "dgcnn_segmentation":
        raise ValueError(f"Unsupported segmentation model: {name!r}")
    return DGCNNSegmentation(
        input_channels=int(model_config.get("input_channels", 6)),
        num_classes=int(model_config.get("num_classes", 33)),
        k=int(model_config.get("k", 20)),
        emb_dims=int(model_config.get("emb_dims", 1024)),
        dropout=float(model_config.get("dropout", 0.5)),
    )


def build_batch_preprocessor(data_config: dict[str, Any], seed: int):
    sampling_config = data_config.get("sampling", {})
    num_points = sampling_config.get("num_points")
    if num_points is None:
        return None
    method = sampling_config.get("method", "random")
    if method != "random":
        raise ValueError(f"Training sampling only supports 'random', got {method!r}")
    sampler_seed = sampling_config.get("seed", seed)
    generator = None
    if sampler_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(sampler_seed))

    def preprocess(batch):
        return random_sample_point_batch(batch, num_points=int(num_points), generator=generator)

    return preprocess


def build_run_dir(experiment_config: dict[str, Any]) -> Path:
    root = resolve_project_path(experiment_config.get("output_dir", "outputs/experiments"))
    if root is None:
        root = PROJECT_ROOT / "outputs" / "experiments"
    run_dir = root / experiment_config.get("name", "experiment")
    return ensure_dir(run_dir)


def build_logger(
    logging_config: dict[str, Any],
    experiment_config: dict[str, Any],
    run_dir: Path,
    config: dict[str, Any],
) -> CompositeLogger:
    loggers = [
        ConsoleLogger(log_every_steps=int(logging_config.get("log_every_steps", 1))),
        JsonlLogger(run_dir / "metrics.jsonl"),
    ]
    if bool(logging_config.get("wandb", False)):
        loggers.append(
            WandbLogger(
                project=logging_config.get("project", "OrthoTwin3D"),
                name=experiment_config.get("name", "experiment"),
                config=config,
            )
        )
    return CompositeLogger(loggers)


def save_config_copy(config: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    main()
