import torch


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Dense confusion matrix with rows=target and columns=prediction."""
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    bins = target * num_classes + pred
    matrix = torch.bincount(bins, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def segmentation_metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    """Return global accuracy, mIoU and mean F1."""
    true_positive = matrix.diag()
    target_count = matrix.sum(dim=1)
    pred_count = matrix.sum(dim=0)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1.0)

    union = target_count + pred_count - true_positive
    valid_iou = union > 0
    iou = true_positive / union.clamp_min(1.0)
    miou = iou[valid_iou].mean() if valid_iou.any() else matrix.new_tensor(0.0)

    f1_denominator = target_count + pred_count
    valid_f1 = f1_denominator > 0
    f1 = (2.0 * true_positive) / f1_denominator.clamp_min(1.0)
    mean_f1 = f1[valid_f1].mean() if valid_f1.any() else matrix.new_tensor(0.0)
    return {
        "accuracy": float(accuracy.detach().cpu()),
        "miou": float(miou.detach().cpu()),
        "mean_f1": float(mean_f1.detach().cpu()),
    }
