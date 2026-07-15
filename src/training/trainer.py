from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Sequence

import torch

from src.training.checkpointing import CheckpointManager, load_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.loggers import BaseLogger, CompositeLogger
from src.training.metrics import segmentation_metrics_from_confusion
from src.training.tasks import Task
from src.training.utils import (
    average_metric_dicts,
    flatten_metrics,
    get_device,
    move_to_device,
)
from src.utils.logger import get_logger


LOGGER = get_logger("training")


class Trainer:
    """Shared train/evaluation loop driven by a task object."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        optimizer: torch.optim.Optimizer,
        train_batch_preprocessor: Callable[..., Any] | None = None,
        eval_batch_preprocessor: Callable[..., Any] | None = None,
        eval_view_ids: Sequence[int | None] | None = None,
        scheduler: Any | None = None,
        scheduler_step_metric: str | None = None,
        device: str | torch.device = "auto",
        logger: BaseLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        early_stopping: EarlyStopping | None = None,
        max_epochs: int = 100,
        grad_clip: float | None = None,
        amp: bool = False,
        validation_every_epochs: int = 1,
        evaluate_epoch_zero: bool = False,
        log_every_epochs: int = 1,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_metric = scheduler_step_metric
        self.logger = logger or CompositeLogger([])
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.max_epochs = int(max_epochs)
        self.grad_clip = grad_clip
        self.train_batch_preprocessor = train_batch_preprocessor
        self.eval_batch_preprocessor = eval_batch_preprocessor
        self.eval_view_ids = (
            list(eval_view_ids) if eval_view_ids is not None else [None]
        )
        self.validation_every_epochs = max(1, int(validation_every_epochs))
        self.evaluate_epoch_zero = bool(evaluate_epoch_zero)
        self.log_every_epochs = max(1, int(log_every_epochs))
        self.config = config or {}
        self.global_step = 0
        self.start_epoch = 1
        self.use_amp = bool(amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def fit(self, train_loader, val_loader=None) -> None:
        try:
            if self.evaluate_epoch_zero and self.start_epoch == 1:
                self.evaluate(train_loader, epoch=0, split="train")
                if val_loader is not None:
                    self.evaluate(val_loader, epoch=0, split="val")

            for epoch in range(self.start_epoch, self.max_epochs + 1):
                should_stop = False
                train_metrics = self.train_epoch(train_loader, epoch)
                epoch_metrics = self._prefix_metrics(train_metrics, "train")
                epoch_metrics["train_lr"] = self._current_lr()
                if self._should_log_epoch(epoch):
                    self.logger.log(
                        epoch_metrics, step=self.global_step, epoch=epoch, split="train"
                    )

                if val_loader is not None and self._should_validate(epoch):
                    val_metrics = self.evaluate(val_loader, epoch=epoch, split="val")
                    epoch_metrics.update(self._prefix_metrics(val_metrics, "val"))
                    if self.early_stopping is not None:
                        should_stop = self.early_stopping.update(
                            epoch_metrics, epoch=epoch
                        )

                self._step_scheduler(epoch_metrics)
                self._save_checkpoints(epoch, epoch_metrics)
                if should_stop:
                    LOGGER.info(
                        "early stopping at epoch=%s | monitor=%s | "
                        "best_epoch=%s | best_metric=%.4f",
                        epoch,
                        self.early_stopping.monitor,
                        self.early_stopping.best_epoch,
                        self.early_stopping.best_metric,
                    )
                    break
        finally:
            self.logger.close()

    def train_epoch(self, train_loader, epoch: int | None = None) -> dict[str, float]:
        self.model.train()
        collected = []
        weights = []
        confusion_matrix = None
        current_epoch = epoch or 0

        for batch in train_loader:
            batch = self._preprocess_batch(
                self.train_batch_preprocessor,
                batch,
                epoch=current_epoch,
                split="train",
            )
            batch = move_to_device(batch, self.device)
            with self._autocast_context():
                output = self.task.training_step(self.model, batch)
                loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={current_epoch} step={self.global_step + 1}"
                )

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            self.global_step += 1
            metrics = flatten_metrics(output.get("metrics", {}))
            collected.append(metrics)
            weights.append(_output_weight(output))
            confusion_matrix = _accumulate_confusion_matrix(
                confusion_matrix, output.get("confusion_matrix")
            )

        return _aggregate_epoch_metrics(collected, weights, confusion_matrix)

    @torch.inference_mode()
    def evaluate(
        self, loader, epoch: int | None = None, split: str = "val"
    ) -> dict[str, float]:
        self.model.eval()
        collected = []
        weights = []
        confusion_matrix = None
        current_epoch = epoch or 0
        for view_id in self.eval_view_ids:
            for batch in loader:
                batch = self._preprocess_batch(
                    self.eval_batch_preprocessor,
                    batch,
                    epoch=current_epoch,
                    split=split,
                    view_id=view_id,
                )
                batch = move_to_device(batch, self.device)
                with self._autocast_context():
                    output = self.task.validation_step(self.model, batch)
                collected.append(flatten_metrics(output.get("metrics", {})))
                weights.append(_output_weight(output))
                confusion_matrix = _accumulate_confusion_matrix(
                    confusion_matrix, output.get("confusion_matrix")
                )

        metrics = _aggregate_epoch_metrics(collected, weights, confusion_matrix)
        metrics["eval_views"] = float(len(self.eval_view_ids))
        self.logger.log(
            self._prefix_metrics(metrics, split),
            step=self.global_step,
            epoch=current_epoch,
            split=split,
        )
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
        if self.early_stopping is not None:
            training_state = checkpoint.get("training_state")
            if not isinstance(training_state, dict):
                raise ValueError("Checkpoint does not contain training_state")
            early_stopping_state = training_state.get("early_stopping")
            if not isinstance(early_stopping_state, dict):
                raise ValueError("Checkpoint does not contain early-stopping state")
            self.early_stopping.load_state_dict(early_stopping_state)
        return checkpoint

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, enabled=True)

    def _step_scheduler(self, metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return
        if self.scheduler_step_metric is None:
            self.scheduler.step()
            return
        if self.scheduler_step_metric in metrics:
            self.scheduler.step(metrics[self.scheduler_step_metric])

    def _current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

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
            training_state=self._training_state(),
        )
        self.checkpoint_manager.save_last(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
            config=self.config,
            training_state=self._training_state(),
        )
        self.checkpoint_manager.save_epoch(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
            config=self.config,
            training_state=self._training_state(),
        )

    def _training_state(self) -> dict[str, Any]:
        if self.early_stopping is None:
            return {}
        return {"early_stopping": self.early_stopping.state_dict()}

    def _should_validate(self, epoch: int) -> bool:
        return (
            _is_scheduled_epoch(epoch, self.validation_every_epochs)
            or epoch == self.max_epochs
        )

    def _should_log_epoch(self, epoch: int) -> bool:
        return (
            _is_scheduled_epoch(epoch, self.log_every_epochs)
            or epoch == self.max_epochs
        )

    @staticmethod
    def _preprocess_batch(
        preprocessor: Callable[..., Any] | None,
        batch: Any,
        epoch: int,
        split: str,
        view_id: int | None = None,
    ) -> Any:
        if preprocessor is None:
            return batch
        return preprocessor(batch, epoch=epoch, split=split, view_id=view_id)

    @staticmethod
    def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _accumulate_confusion_matrix(
    total: torch.Tensor | None, value: Any
) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return total
    value = value.detach().cpu()
    return value if total is None else total + value


def _is_scheduled_epoch(epoch: int, every_epochs: int) -> bool:
    return (epoch - 1) % every_epochs == 0


def _output_weight(output: dict[str, Any]) -> float:
    return float(output.get("num_points", 1.0))


def _aggregate_epoch_metrics(
    collected: list[dict[str, float]],
    weights: list[float],
    confusion_matrix: torch.Tensor | None,
) -> dict[str, float]:
    averaged = average_metric_dicts(collected, weights)
    if confusion_matrix is None:
        return averaged

    metrics = dict(averaged)
    metrics.update(segmentation_metrics_from_confusion(confusion_matrix.float()))
    return metrics
