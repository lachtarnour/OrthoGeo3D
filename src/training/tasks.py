from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from src.training.losses import SegmentationLoss
from src.training.metrics import confusion_matrix, segmentation_metrics_from_confusion


StepOutput = dict[str, Any]


class Task(ABC):
    """Task-specific behavior consumed by the shared Trainer."""

    @abstractmethod
    def training_step(
        self, model: torch.nn.Module, batch: dict[str, Any]
    ) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self, model: torch.nn.Module, batch: dict[str, Any]
    ) -> StepOutput:
        raise NotImplementedError


class SegmentationTask(Task):
    """Loss and metrics for multiclass point segmentation."""

    def __init__(
        self,
        num_classes: int,
        loss_config: Mapping[str, Any],
    ) -> None:
        self.num_classes = int(num_classes)
        self.loss_fn = SegmentationLoss(
            num_classes=self.num_classes,
            config=loss_config,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
    ) -> StepOutput:
        return self._step(model, batch)

    def validation_step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
    ) -> StepOutput:
        return self._step(model, batch)

    def _step(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
    ) -> StepOutput:
        outputs = normalize_model_outputs(model(batch["x"]))
        logits = outputs["logits"]
        loss, loss_metrics = self.loss_fn(logits, batch["y"])
        matrix = confusion_matrix(
            logits.detach().argmax(dim=-1),
            batch["y"].detach(),
            num_classes=self.num_classes,
        )
        metrics = segmentation_metrics_from_confusion(matrix.float())
        metrics.update(loss_metrics)
        metrics["loss"] = float(loss.detach().cpu().item())
        result = {
            "loss": loss,
            "metrics": metrics,
            "logits": logits,
            "confusion_matrix": matrix,
            "num_points": int(batch["y"].numel()),
        }
        result.update({key: value for key, value in outputs.items() if key != "logits"})
        return result


def normalize_model_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    """Accept tensor baselines and dictionary outputs from future multi-task models."""
    if torch.is_tensor(outputs):
        return {"logits": outputs}
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if not torch.is_tensor(logits):
            raise ValueError("Model output dictionary must contain tensor 'logits'")
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")
