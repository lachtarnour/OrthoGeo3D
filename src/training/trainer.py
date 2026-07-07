from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch

from src.training.checkpointing import CheckpointManager, load_checkpoint
from src.training.loggers import BaseLogger, CompositeLogger
from src.training.tasks import Task
from src.training.utils import average_metric_dicts, flatten_metrics, get_device, move_to_device


class Trainer:
    """Generic train/eval loop driven by a task object."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        device: str | torch.device = "auto",
        logger: BaseLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        max_epochs: int = 100,
        grad_clip: float | None = None,
        amp: bool = False,
        batch_preprocessor: Callable[[Any], Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger or CompositeLogger([])
        self.checkpoint_manager = checkpoint_manager
        self.max_epochs = int(max_epochs)
        self.grad_clip = grad_clip
        self.batch_preprocessor = batch_preprocessor
        self.config = config or {}
        self.global_step = 0
        self.start_epoch = 1
        self.use_amp = bool(amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def fit(self, train_loader, val_loader=None) -> None:
        try:
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                train_metrics = self.train_epoch(train_loader, epoch)
                epoch_metrics = self._prefix_metrics(train_metrics, "train")
                self.logger.log(epoch_metrics, step=self.global_step, epoch=epoch, split="train")

                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader, epoch=epoch, split="val")
                    epoch_metrics.update(self._prefix_metrics(val_metrics, "val"))

                self._step_scheduler(epoch_metrics)
                self._save_checkpoints(epoch, epoch_metrics)
        finally:
            self.logger.close()

    def train_epoch(self, train_loader, epoch: int | None = None) -> dict[str, float]:
        self.model.train()
        collected = []
        current_epoch = epoch or 0

        for batch in train_loader:
            if self.batch_preprocessor is not None:
                batch = self.batch_preprocessor(batch)
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_context():
                output = self.task.training_step(self.model, batch)
                loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={current_epoch} step={self.global_step + 1}"
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.global_step += 1
            metrics = flatten_metrics(output.get("metrics", {}))
            collected.append(metrics)
            self.logger.log(metrics, step=self.global_step, epoch=current_epoch, split="train_step")

        return average_metric_dicts(collected)

    @torch.no_grad()
    def evaluate(self, loader, epoch: int | None = None, split: str = "val") -> dict[str, float]:
        self.model.eval()
        collected = []
        current_epoch = epoch or 0

        for batch in loader:
            if self.batch_preprocessor is not None:
                batch = self.batch_preprocessor(batch)
            batch = move_to_device(batch, self.device)
            with self._autocast_context():
                output = self.task.validation_step(self.model, batch)
            collected.append(flatten_metrics(output.get("metrics", {})))

        metrics = average_metric_dicts(collected)
        self.logger.log(self._prefix_metrics(metrics, split), step=self.global_step, epoch=current_epoch, split=split)
        return metrics

    def resume_from_checkpoint(self, path: str) -> dict[str, Any]:
        checkpoint = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.global_step = int(checkpoint.get("step", 0))
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.best_metric = checkpoint.get("best_metric")
        return checkpoint

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, enabled=True)

    def _step_scheduler(self, metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor = self.checkpoint_manager.monitor if self.checkpoint_manager else "val_loss"
            if monitor in metrics:
                self.scheduler.step(metrics[monitor])
            return
        self.scheduler.step()

    def _save_checkpoints(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.checkpoint_manager is None:
            return
        self.checkpoint_manager.save_best(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
            config=self.config,
        )
        self.checkpoint_manager.save_last(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
            config=self.config,
        )
        self.checkpoint_manager.save_epoch(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
            config=self.config,
        )

    @staticmethod
    def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
