#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from src.augmentations import build_train_augmentation
from src.datasets.teeth3ds_processed import (
    Teeth3DSSegmentationDataset,
    create_segmentation_dataloader,
)
from src.models import DGCNNSegmentation
from src.training import (
    CheckpointManager,
    EarlyStopping,
    SegmentationTask,
    Trainer,
    checkpoint_configs,
)
from src.training.loggers import (
    CompositeLogger,
    ConsoleLogger,
    JsonlLogger,
    WandbLogger,
)
from src.training.sampling import build_sampling_preprocessors, eval_view_ids
from src.training.utils import get_device, set_seed
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.paths import get_output_dir, resolve_project_path


DEFAULT_CONFIG_PATH = "configs/train/dgcnn_segmentation.yaml"
DEFAULT_DATA_CONFIG_PATH = "configs/data.yaml"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = load_config(args.data_config)
    resume_path = resolve_project_path(args.resume) if args.resume else None
    if args.resume:
        if resume_path is None or not resume_path.is_file():
            raise FileNotFoundError(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        config, data_config = checkpoint_configs(checkpoint)
    run_config = config["run"]
    model_config = config["model"]
    training_config = config["training"]
    sampling_config = config["sampling"]
    max_epochs = int(training_config["epochs"])

    seed = int(run_config["seed"])
    set_seed(seed)
    device = get_device("auto")
    configure_cuda_performance(config.get("performance", {}), device)

    run_dir = ensure_dir(get_output_dir() / str(run_config["name"]))
    save_config_copy(config, run_dir / "config.yaml")
    train_loader = create_segmentation_dataloader(data_config, split="train")
    val_loader = create_segmentation_dataloader(data_config, split="val")
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, Teeth3DSSegmentationDataset):
        raise TypeError("Expected a Teeth3DSSegmentationDataset")
    validate_model_dataset_contract(model_config, train_dataset)

    model = build_model(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler, scheduler_step_metric = build_scheduler(
        optimizer,
        scheduler_config=config["scheduler"],
        monitor=str(training_config["monitor"]),
        mode=str(training_config["monitor_mode"]),
    )
    task = SegmentationTask(
        num_classes=int(model_config["num_classes"]),
        loss_config=config["loss"],
    )
    logger = build_logger(
        config["logging"],
        run_config,
        run_dir,
        config,
    )
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_dir / "checkpoints",
        monitor=str(training_config["monitor"]),
        mode=str(training_config["monitor_mode"]),
        save_every_epochs=int(config["checkpoint"]["save_every_epochs"]),
    )
    early_stopping = build_early_stopping(
        config["early_stopping"],
        monitor=str(training_config["monitor"]),
        mode=str(training_config["monitor_mode"]),
    )
    train_augmentation = build_train_augmentation(
        config.get("augmentation"),
        seed=seed,
        feature_keys=train_dataset.feature_keys,
    )
    train_preprocessor, eval_preprocessor = build_sampling_preprocessors(
        sampling_config,
        seed=seed,
        train_transform=train_augmentation,
    )

    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_metric=scheduler_step_metric,
        device=device,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        max_epochs=max_epochs,
        grad_clip=float(training_config["grad_clip"]),
        amp=bool(training_config["amp"]),
        train_batch_preprocessor=train_preprocessor,
        eval_batch_preprocessor=eval_preprocessor,
        eval_view_ids=eval_view_ids(sampling_config),
        validation_every_epochs=int(training_config["validation_every_epochs"]),
        evaluate_epoch_zero=bool(training_config["evaluate_epoch_zero"]),
        log_every_epochs=int(training_config["log_every_epochs"]),
        config={"train_config": config, "data_config": data_config},
    )

    if resume_path is not None:
        trainer.resume_from_checkpoint(str(resume_path))
    trainer.fit(train_loader, val_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DGCNN tooth segmentation model."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG_PATH)
    parser.add_argument("--resume", help="Checkpoint path to resume from.")
    return parser.parse_args()


def configure_cuda_performance(
    performance_config: dict[str, Any],
    device: torch.device,
) -> None:
    if device.type != "cuda":
        return
    use_tf32 = bool(performance_config.get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if use_tf32:
        torch.set_float32_matmul_precision(
            str(performance_config.get("float32_matmul_precision", "high"))
        )


def build_model(model_config: dict[str, Any]) -> DGCNNSegmentation:
    return DGCNNSegmentation(
        input_channels=int(model_config["input_channels"]),
        num_classes=int(model_config["num_classes"]),
        k=int(model_config["k"]),
        emb_dims=int(model_config["emb_dims"]),
        dropout=float(model_config["dropout"]),
    )


def build_early_stopping(
    config: dict[str, Any],
    monitor: str,
    mode: str,
) -> EarlyStopping:
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=int(config["patience"]),
        min_delta=float(config["min_delta"]),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    monitor: str,
    mode: str,
) -> tuple[Any, str]:
    name = str(scheduler_config["name"])
    if name != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler: {name!r}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        threshold=float(scheduler_config["threshold"]),
        threshold_mode=str(scheduler_config["threshold_mode"]),
        min_lr=float(scheduler_config["min_lr"]),
    )
    return scheduler, monitor


def validate_model_dataset_contract(
    model_config: dict[str, Any],
    dataset: Teeth3DSSegmentationDataset,
) -> None:
    if int(model_config["input_channels"]) != dataset.feature_dim:
        raise ValueError(
            f"Model expects {model_config['input_channels']} channels, "
            f"dataset provides {dataset.feature_dim}"
        )
    if int(model_config["num_classes"]) != dataset.num_classes:
        raise ValueError(
            f"Model expects {model_config['num_classes']} classes, "
            f"dataset provides {dataset.num_classes}"
        )


def build_logger(
    logging_config: dict[str, Any],
    run_config: dict[str, Any],
    run_dir: Path,
    config: dict[str, Any],
) -> CompositeLogger:
    loggers = [ConsoleLogger(), JsonlLogger(run_dir / "metrics.jsonl")]
    if bool(logging_config.get("wandb", False)):
        loggers.append(
            WandbLogger(
                project=str(logging_config["project"]),
                name=str(run_config["name"]),
                config=config,
            )
        )
    return CompositeLogger(loggers)


def save_config_copy(config: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
