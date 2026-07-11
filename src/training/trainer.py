from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Sequence

import torch

from src.training.checkpointing import CheckpointManager, load_checkpoint
from src.training.loggers import BaseLogger, CompositeLogger
from src.training.metrics import segmentation_metrics_from_confusion
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
        train_batch_preprocessor: Callable[[Any], Any] | None = None,
        eval_batch_preprocessor: Callable[[Any], Any] | None = None,
        eval_view_ids: Sequence[int] | None = None,
        validation_every_epochs: int = 1,
        validate_last_epoch: bool = True,
        log_every_epochs: int = 1,
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
        self.train_batch_preprocessor = train_batch_preprocessor
        self.eval_batch_preprocessor = eval_batch_preprocessor
        self.eval_view_ids = list(eval_view_ids) if eval_view_ids is not None else None
        self.validation_every_epochs = max(1, int(validation_every_epochs))
        self.validate_last_epoch = bool(validate_last_epoch)
        self.log_every_epochs = max(1, int(log_every_epochs))
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
                epoch_metrics["train_lr"] = self._current_lr()
                if self._should_log_epoch(epoch):
                    self.logger.log(epoch_metrics, step=self.global_step, epoch=epoch, split="train")

                if val_loader is not None and self._should_validate(epoch):
                    val_metrics = self.evaluate(val_loader, epoch=epoch, split="val")
                    epoch_metrics.update(self._prefix_metrics(val_metrics, "val"))

                self._step_scheduler()
                self._save_checkpoints(epoch, epoch_metrics)
        finally:
            self.logger.close()

    def train_epoch(self, train_loader, epoch: int | None = None) -> dict[str, float]:
        self.model.train()
        collected = []
        confusion_matrix = None
        current_epoch = epoch or 0

        for batch_index, batch in enumerate(train_loader, start=1):
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
            confusion_matrix = _accumulate_confusion_matrix(confusion_matrix, output.get("confusion_matrix"))
            self.logger.log(metrics, step=self.global_step, epoch=current_epoch, split="train_step")

        return _aggregate_epoch_metrics(collected, confusion_matrix)

    @torch.inference_mode()
    def evaluate(self, loader, epoch: int | None = None, split: str = "val") -> dict[str, float]:
        self.model.eval()
        collected = []
        confusion_matrix = None
        current_epoch = epoch or 0
        view_ids = self.eval_view_ids if self.eval_view_ids is not None else [None]

        for view_id in view_ids:
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
                confusion_matrix = _accumulate_confusion_matrix(confusion_matrix, output.get("confusion_matrix"))

        metrics = _aggregate_epoch_metrics(collected, confusion_matrix)
        if self.eval_view_ids is not None:
            metrics["eval_views"] = float(len(view_ids))
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

    def _step_scheduler(self) -> None:
        if self.scheduler is None:
            return
        self.scheduler.step()

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

    def _should_validate(self, epoch: int) -> bool:
        if epoch % self.validation_every_epochs == 0:
            return True
        return self.validate_last_epoch and epoch == self.max_epochs

    def _should_log_epoch(self, epoch: int) -> bool:
        if epoch % self.log_every_epochs == 0:
            return True
        return self.validate_last_epoch and epoch == self.max_epochs

    @staticmethod
    def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    @staticmethod
    def _preprocess_batch(
        preprocessor: Callable[[Any], Any] | None,
        batch: Any,
        epoch: int,
        split: str,
        view_id: int | None = None,
    ) -> Any:
        if preprocessor is None:
            return batch
        return preprocessor(batch, epoch=epoch, split=split, view_id=view_id)


def _accumulate_confusion_matrix(total: torch.Tensor | None, value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return total
    value = value.detach().cpu()
    return value if total is None else total + value


def _aggregate_epoch_metrics(
    collected: list[dict[str, float]],
    confusion_matrix: torch.Tensor | None,
) -> dict[str, float]:
    averaged = average_metric_dicts(collected)
    if confusion_matrix is None:
        return averaged

    metrics = segmentation_metrics_from_confusion(confusion_matrix.float())
    if "loss" in averaged:
        metrics["loss"] = averaged["loss"]
    return metrics
