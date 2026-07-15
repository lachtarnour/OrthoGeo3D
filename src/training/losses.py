from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


class SegmentationLoss:
    """Composite loss for multiclass tooth segmentation."""

    def __init__(
        self,
        num_classes: int,
        config: Mapping[str, Any],
    ) -> None:
        self.num_classes = int(num_classes)
        self.cross_entropy_weight = float(config["cross_entropy_weight"])
        self.dice_weight = float(config["dice_weight"])
        self.binary_weight = float(config["binary_weight"])
        self.max_cross_entropy = float(config["max_cross_entropy"])
        self.max_binary = float(config["max_binary"])

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        validate_segmentation_tensors(logits, target, self.num_classes)
        cross_entropy = cross_entropy_loss(
            logits,
            target,
            max_loss=self.max_cross_entropy,
        )
        dice = dice_loss(logits, target, include_background=False)
        binary = binary_tooth_loss(
            logits,
            target,
            max_loss=self.max_binary,
        )
        total = (
            self.cross_entropy_weight * cross_entropy
            + self.dice_weight * dice
            + self.binary_weight * binary
        )
        return total, {
            "loss_cross_entropy": float(cross_entropy.detach().cpu()),
            "loss_dice": float(dice.detach().cpu()),
            "loss_binary": float(binary.detach().cpu()),
        }


def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_loss: float,
) -> torch.Tensor:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    )
    return loss.clamp(max=max_loss).mean()


def binary_tooth_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    max_loss: float,
) -> torch.Tensor:
    background_logit = logits[..., 0]
    tooth_logit = torch.logsumexp(logits[..., 1:], dim=-1)
    binary_logits = torch.stack((background_logit, tooth_logit), dim=-1)
    binary_target = (target > 0).long()
    loss = F.cross_entropy(
        binary_logits.reshape(-1, 2),
        binary_target.reshape(-1),
        reduction="none",
    )
    return loss.clamp(max=max_loss).mean()


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    include_background: bool,
    smooth: float = 1.0,
) -> torch.Tensor:
    num_classes = logits.shape[-1]
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(target, num_classes=num_classes).to(probabilities.dtype)
    start = 0 if include_background else 1
    probabilities = probabilities[..., start:]
    one_hot = one_hot[..., start:]

    present = one_hot.sum(dim=(0, 1)) > 0
    if not present.any():
        return logits.sum() * 0.0
    probabilities = probabilities[..., present]
    one_hot = one_hot[..., present]
    intersection = (probabilities * one_hot).sum(dim=(0, 1))
    denominator = probabilities.sum(dim=(0, 1)) + one_hot.sum(dim=(0, 1))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def validate_segmentation_tensors(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    if logits.ndim != 3 or logits.shape[-1] != num_classes:
        raise ValueError(
            f"Expected logits with shape [B, N, {num_classes}], "
            f"got {tuple(logits.shape)}"
        )
    if target.shape != logits.shape[:2]:
        raise ValueError(
            f"Expected target with shape {tuple(logits.shape[:2])}, "
            f"got {tuple(target.shape)}"
        )
    invalid = (target < 0) | (target >= num_classes)
    if invalid.any():
        values = sorted(set(target[invalid].detach().cpu().tolist()))
        raise ValueError(
            f"Target contains labels outside [0, {num_classes - 1}]: {values}"
        )
