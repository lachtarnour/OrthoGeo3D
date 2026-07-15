from pathlib import Path
from typing import Any, Literal

import torch

from src.utils.io import ensure_dir


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_every_epochs: int = 1,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_every_epochs = max(1, int(save_every_epochs))
        self.best_metric: float | None = None

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        config: dict[str, Any],
        training_state: dict[str, Any] | None = None,
    ) -> Path:
        return self._save(
            "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            metrics,
            config,
            training_state,
        )

    def save_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        config: dict[str, Any],
        training_state: dict[str, Any] | None = None,
    ) -> Path | None:
        if epoch % self.save_every_epochs != 0:
            return None
        return self._save(
            f"epoch_{epoch:04d}.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            metrics,
            config,
            training_state,
        )

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        config: dict[str, Any],
        training_state: dict[str, Any] | None = None,
    ) -> Path | None:
        if self.monitor not in metrics:
            return None
        value = float(metrics[self.monitor])
        if not self.is_better(value):
            return None
        self.best_metric = value
        return self._save(
            "best.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            step,
            metrics,
            config,
            training_state,
        )

    def is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return value < self.best_metric
        return value > self.best_metric

    def _save(
        self,
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        config: dict[str, Any],
        training_state: dict[str, Any] | None,
    ) -> Path:
        path = self.checkpoint_dir / filename
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if scheduler is not None
            else None,
            "epoch": int(epoch),
            "step": int(step),
            "best_metric": self.best_metric,
            "metrics": metrics,
            "config": config,
            "training_state": training_state or {},
        }

        torch.save(payload, path)
        return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def checkpoint_configs(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the complete reproducible configuration embedded in a checkpoint."""
    embedded = checkpoint.get("config")
    if not isinstance(embedded, dict):
        raise ValueError("Checkpoint does not contain an embedded configuration")

    train_config = embedded.get("train_config")
    data_config = embedded.get("data_config")
    if not isinstance(train_config, dict) or not isinstance(data_config, dict):
        raise ValueError(
            "Checkpoint configuration must contain train_config and data_config"
        )
    return train_config, data_config
