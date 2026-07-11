from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


class SegmentationLoss:
    """Composite point-wise segmentation loss for arch-normalized tooth labels."""

    def __init__(
        self,
        num_classes: int,
        config: Mapping[str, Any] | None = None,
        ignore_index: int | None = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.components = _loss_components(config or {})

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        _validate_logits(logits, self.num_classes)
        total = logits.sum() * 0.0
        metrics: dict[str, float] = {}
        enabled = 0

        ce_config = self.components["cross_entropy"]
        if ce_config["enabled"]:
            value = cross_entropy_loss(
                logits,
                target,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                max_loss=ce_config.get("max_loss"),
            )
            total = total + float(ce_config.get("weight", 1.0)) * value
            metrics["loss_cross_entropy"] = _to_float(value)
            enabled += 1

        dice_config = self.components["dice"]
        if dice_config["enabled"]:
            value = dice_loss(
                logits,
                target,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                include_background=bool(dice_config.get("include_background", False)),
                smooth=float(dice_config.get("smooth", 1.0)),
            )
            total = total + float(dice_config.get("weight", 1.0)) * value
            metrics["loss_dice"] = _to_float(value)
            enabled += 1

        binary_config = self.components["binary"]
        if binary_config["enabled"]:
            value = binary_tooth_loss(
                logits,
                target,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                max_loss=binary_config.get("max_loss"),
            )
            total = total + float(binary_config.get("weight", 1.0)) * value
            metrics["loss_binary"] = _to_float(value)
            enabled += 1

        if enabled == 0:
            raise ValueError("At least one segmentation loss component must be enabled")
        return total, metrics


def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    max_loss: float | int | None = None,
) -> torch.Tensor:
    flat_logits, flat_target = _flatten_valid_targets(logits, target, num_classes, ignore_index)
    if flat_target.numel() == 0:
        return logits.sum() * 0.0

    loss = F.cross_entropy(flat_logits, flat_target, reduction="none")
    if max_loss is not None:
        loss = loss.clamp(max=float(max_loss))
    return loss.mean()


def binary_tooth_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    max_loss: float | int | None = None,
) -> torch.Tensor:
    if num_classes < 2:
        raise ValueError("Binary tooth/background loss requires at least two semantic classes")

    background_logit = logits[..., 0]
    tooth_logit = torch.logsumexp(logits[..., 1:], dim=-1)
    binary_logits = torch.stack([background_logit, tooth_logit], dim=-1)
    binary_target = (target > 0).long()
    flat_logits, flat_target = _flatten_valid_targets(
        binary_logits,
        binary_target,
        num_classes=2,
        ignore_index=ignore_index,
        validity_source=target,
        validity_num_classes=num_classes,
    )
    if flat_target.numel() == 0:
        return logits.sum() * 0.0

    loss = F.cross_entropy(flat_logits, flat_target, reduction="none")
    if max_loss is not None:
        loss = loss.clamp(max=float(max_loss))
    return loss.mean()


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    include_background: bool = False,
    smooth: float = 1.0,
) -> torch.Tensor:
    valid_mask = _valid_target_mask(target, num_classes, ignore_index)
    if not valid_mask.any():
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes)
    one_hot = one_hot.to(device=logits.device, dtype=probs.dtype)

    valid = valid_mask.to(device=logits.device).unsqueeze(-1)
    probs = probs * valid
    one_hot = one_hot * valid

    start = 0 if include_background else 1
    probs = probs[..., start:]
    one_hot = one_hot[..., start:]
    if probs.shape[-1] == 0:
        return logits.sum() * 0.0

    present = one_hot.sum(dim=(0, 1)) > 0
    if not present.any():
        return logits.sum() * 0.0

    probs = probs[..., present]
    one_hot = one_hot[..., present]
    intersection = (probs * one_hot).sum(dim=(0, 1))
    denominator = probs.sum(dim=(0, 1)) + one_hot.sum(dim=(0, 1))
    dice = (2.0 * intersection + float(smooth)) / (denominator + float(smooth)).clamp_min(1.0e-8)
    return 1.0 - dice.mean()


def _loss_components(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    components = config.get("components")
    if components is None:
        return {
            "cross_entropy": {"enabled": True, "weight": 1.0, "max_loss": None},
            "dice": {"enabled": False},
            "binary": {"enabled": False},
        }
    if not isinstance(components, Mapping):
        raise ValueError("loss.components must be a mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for name in ("cross_entropy", "dice", "binary"):
        raw = components.get(name, False)
        if isinstance(raw, bool):
            normalized[name] = {"enabled": raw}
        elif isinstance(raw, Mapping):
            normalized[name] = dict(raw)
            normalized[name]["enabled"] = bool(normalized[name].get("enabled", True))
        else:
            raise ValueError(f"loss.components.{name} must be a mapping or boolean")
    return normalized


def _flatten_valid_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None,
    validity_source: torch.Tensor | None = None,
    validity_num_classes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = target if validity_source is None else validity_source
    source_classes = num_classes if validity_num_classes is None else validity_num_classes
    valid_mask = _valid_target_mask(source, source_classes, ignore_index)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1).to(device=logits.device)
    flat_mask = valid_mask.reshape(-1).to(device=logits.device)
    return flat_logits[flat_mask], flat_target[flat_mask]


def _valid_target_mask(target: torch.Tensor, num_classes: int, ignore_index: int | None) -> torch.Tensor:
    target = target.to(dtype=torch.long)
    valid_mask = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid_mask = target != int(ignore_index)

    invalid = valid_mask & ((target < 0) | (target >= int(num_classes)))
    if invalid.any():
        invalid_values = sorted(set(target[invalid].detach().cpu().reshape(-1).tolist()))
        raise ValueError(f"Target contains labels outside [0, {num_classes - 1}]: {invalid_values}")
    return valid_mask


def _validate_logits(logits: torch.Tensor, num_classes: int) -> None:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, N, C], got {tuple(logits.shape)}")
    if logits.shape[-1] != num_classes:
        raise ValueError(f"Expected {num_classes} classes, got logits shape {tuple(logits.shape)}")


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
