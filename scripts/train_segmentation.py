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
    create_segmentation_dataloader,
)
from src.models import DGCNNSegmentation
from src.training import CheckpointManager, SegmentationTask, Trainer
from src.training.loggers import CompositeLogger, ConsoleLogger, JsonlLogger, WandbLogger
from src.training.sampling import build_sampling_preprocessors, eval_view_ids_from_config
from src.training.utils import ensure_dir, get_device, set_seed
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_project_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_config = config.get("experiment", {})
    data_config = config.get("data", {})
    optim_config = config.get("optim", {})
    performance_config = config.get("performance", {})
    validation_config = config.get("validation", {})
    logging_config = config.get("logging", {})
    checkpoint_config = config.get("checkpoint", {})
    scheduler_config = optim_config.get("scheduler") or {}
    max_epochs = int(optim_config.get("epochs", 100))

    seed = int(experiment_config.get("seed", 42))
    set_seed(seed)
    device = get_device(optim_config.get("device", "auto"))
    configure_cuda_performance(performance_config, device=device)

    run_dir = build_run_dir(experiment_config)
    save_config_copy(config, run_dir / "config.yaml")

    train_loader = create_segmentation_dataloader(
        data_config.get("config_path", "configs/data.yaml"),
        split=data_config.get("train_split", "train"),
        limit=data_config.get("train_limit"),
        dataloader_config_override=data_config.get("dataloader"),
    )
    val_loader = create_segmentation_dataloader(
        data_config.get("config_path", "configs/data.yaml"),
        split=data_config.get("val_split", "val"),
        limit=data_config.get("val_limit"),
        dataloader_config_override=data_config.get("dataloader"),
    )

    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, Teeth3DSSegmentationDataset):
        raise TypeError("train_loader.dataset must be a Teeth3DSSegmentationDataset")

    model_config = dict(config.get("model", {}))
    model_config.setdefault("input_channels", train_dataset.feature_dim)
    model_config.setdefault("num_classes", train_dataset.num_classes)
    validate_model_dataset_contract(model_config, train_dataset)

    model = build_model(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_config.get("lr", 1.0e-3)),
        weight_decay=float(optim_config.get("weight_decay", 1.0e-4)),
    )
    scheduler = build_scheduler(scheduler_config, optimizer, max_epochs=max_epochs)
    num_classes = int(model_config["num_classes"])

    task = SegmentationTask(
        num_classes=num_classes,
        loss_config=config.get("loss", {}),
    )
    logger = build_logger(logging_config, experiment_config, run_dir, config)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=run_dir / "checkpoints",
        monitor=checkpoint_config.get("monitor", "val_loss"),
        mode=checkpoint_config.get("mode", "min"),
        save_every_epochs=int(checkpoint_config.get("save_every_epochs", 1)),
    )
    train_batch_preprocessor, eval_batch_preprocessor = build_sampling_preprocessors(data_config, seed=seed)
    eval_view_ids = eval_view_ids_from_config(data_config)

    trainer = Trainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        max_epochs=max_epochs,
        grad_clip=optim_config.get("grad_clip"),
        amp=bool(optim_config.get("amp", False)),
        train_batch_preprocessor=train_batch_preprocessor,
        eval_batch_preprocessor=eval_batch_preprocessor,
        eval_view_ids=eval_view_ids,
        validation_every_epochs=int(validation_config.get("every_epochs", 1)),
        validate_last_epoch=bool(validation_config.get("last_epoch", True)),
        log_every_epochs=int(logging_config.get("every_epochs", 1)),
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


def configure_cuda_performance(performance_config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return

    use_tf32 = bool(performance_config.get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if use_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(performance_config.get("float32_matmul_precision", "high")))

    torch.backends.cudnn.benchmark = bool(performance_config.get("cudnn_benchmark", False))


def build_scheduler(
    scheduler_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 100,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(scheduler_config.get("name", "none")).lower() if scheduler_config else "none"
    if name in {None, "none", "off"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(scheduler_config.get("t_max", max_epochs))),
        eta_min=float(scheduler_config.get("min_lr", 0.0)),
    )


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    name = model_config.get("name", "dgcnn_segmentation")
    if name != "dgcnn_segmentation":
        raise ValueError(f"Unsupported segmentation model: {name!r}")
    return DGCNNSegmentation(
        input_channels=int(model_config.get("input_channels", 6)),
        num_classes=int(model_config.get("num_classes", 17)),
        k=int(model_config.get("k", 20)),
        emb_dims=int(model_config.get("emb_dims", 1024)),
        dropout=float(model_config.get("dropout", 0.5)),
    )


def validate_model_dataset_contract(
    model_config: dict[str, Any],
    dataset: Teeth3DSSegmentationDataset,
) -> None:
    input_channels = int(model_config["input_channels"])
    if input_channels != dataset.feature_dim:
        raise ValueError(
            f"model.input_channels={input_channels} does not match dataset feature_dim={dataset.feature_dim} "
            f"for feature_keys={list(dataset.feature_keys)}"
        )

    num_classes = int(model_config["num_classes"])
    if num_classes != dataset.num_classes:
        raise ValueError(
            f"model.num_classes={num_classes} does not match target {dataset.target_key!r} "
            f"with {dataset.num_classes} classes"
        )


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
    log_every_steps = int(logging_config.get("log_every_steps", 1))
    log_train_steps = bool(logging_config.get("log_train_steps", True))
    loggers = [
        ConsoleLogger(log_every_steps=log_every_steps, log_train_steps=log_train_steps),
        JsonlLogger(run_dir / "metrics.jsonl", log_train_steps=log_train_steps),
    ]
    if bool(logging_config.get("wandb", False)):
        loggers.append(
            WandbLogger(
                project=logging_config.get("project", "OrthoTwin3D"),
                name=experiment_config.get("name", "experiment"),
                config=config,
                log_every_steps=log_every_steps,
                log_train_steps=log_train_steps,
            )
        )
    return CompositeLogger(loggers)


def save_config_copy(config: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


if __name__ == "__main__":
    main()
