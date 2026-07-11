from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from src.training.losses import SegmentationLoss
from src.training.metrics import confusion_matrix, segmentation_metrics_from_confusion


StepOutput = dict[str, Any]


class Task(ABC):
    """Task-specific training logic consumed by the generic Trainer."""

    @abstractmethod
    def training_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        raise NotImplementedError


class SegmentationTask(Task):
    """Point-wise tooth segmentation task."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        loss_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.loss_fn = SegmentationLoss(num_classes=self.num_classes, config=loss_config, ignore_index=ignore_index)

    def training_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        return self._step(model, batch)

    def validation_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        return self._step(model, batch)

    def _step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        x = batch["x"]
        y = batch["y"]

        outputs = model(x)
        logits = normalize_model_outputs(outputs)["logits"]
        loss, loss_metrics = self.loss_fn(logits, y)
        pred = logits.detach().argmax(dim=-1)
        matrix = confusion_matrix(
            pred,
            y.detach(),
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        metrics = segmentation_metrics_from_confusion(matrix.float())
        metrics.update(loss_metrics)
        metrics["loss"] = float(loss.detach().cpu().item())
        return {"loss": loss, "metrics": metrics, "logits": logits, "confusion_matrix": matrix}


def normalize_model_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    """Normalize current tensor-only models and future dict-output models."""
    if torch.is_tensor(outputs):
        return {"logits": outputs}
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict must contain a 'logits' key for SegmentationTask")
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")
