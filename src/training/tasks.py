from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

from src.training.metrics import segmentation_metrics


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
    """FDI point-wise segmentation task."""

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        ignore_index: int | None = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_weights = class_weights.float() if class_weights is not None else None
        self.ignore_index = ignore_index

    def training_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        return self._step(model, batch)

    def validation_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        return self._step(model, batch)

    def _step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        x = batch["x"]
        y = batch["y"]

        outputs = model(x)
        logits = normalize_model_outputs(outputs)["logits"]
        loss = self._cross_entropy(logits, y)
        metrics = segmentation_metrics(
            logits.detach(),
            y.detach(),
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        metrics["loss"] = float(loss.detach().cpu().item())
        return {"loss": loss, "metrics": metrics, "logits": logits}

    def _cross_entropy(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape [B, N, C], got {tuple(logits.shape)}")
        if logits.shape[-1] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got logits shape {tuple(logits.shape)}")

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        kwargs = {"weight": weight}
        if self.ignore_index is not None:
            kwargs["ignore_index"] = self.ignore_index
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            **kwargs,
        )


def normalize_model_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    """Normalize current tensor-only models and future dict-output models."""
    if torch.is_tensor(outputs):
        return {"logits": outputs}
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict must contain a 'logits' key for SegmentationTask")
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")
