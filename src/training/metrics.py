import torch


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Compute a dense confusion matrix with rows=target and columns=pred."""
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()

    mask = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    if ignore_index is not None:
        mask = mask & (target != ignore_index)

    pred = pred[mask]
    target = target[mask]
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long, device=pred.device)

    bins = target * num_classes + pred
    matrix = torch.bincount(bins, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Return global accuracy, mIoU and mean F1 for point-wise logits."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, N, C], got {tuple(logits.shape)}")
    if target.shape != logits.shape[:2]:
        raise ValueError(f"Expected target with shape {tuple(logits.shape[:2])}, got {tuple(target.shape)}")

    pred = logits.argmax(dim=-1)
    matrix = confusion_matrix(pred, target, num_classes=num_classes, ignore_index=ignore_index).float()
    return segmentation_metrics_from_confusion(matrix)


def segmentation_metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    """Return accuracy, mIoU and mean F1 from an accumulated confusion matrix."""
    true_positive = matrix.diag()
    target_count = matrix.sum(dim=1)
    pred_count = matrix.sum(dim=0)
    total = matrix.sum().clamp_min(1.0)

    accuracy = true_positive.sum() / total
    union = target_count + pred_count - true_positive
    valid_iou = union > 0
    iou = true_positive / union.clamp_min(1.0)
    miou = iou[valid_iou].mean() if valid_iou.any() else torch.tensor(0.0, device=matrix.device)

    f1_denominator = target_count + pred_count
    valid_f1 = f1_denominator > 0
    f1 = (2.0 * true_positive) / f1_denominator.clamp_min(1.0)
    mean_f1 = f1[valid_f1].mean() if valid_f1.any() else torch.tensor(0.0, device=matrix.device)

    return {
        "accuracy": float(accuracy.detach().cpu().item()),
        "miou": float(miou.detach().cpu().item()),
        "mean_f1": float(mean_f1.detach().cpu().item()),
    }
