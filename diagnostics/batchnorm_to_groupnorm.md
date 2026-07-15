# Replacing BatchNorm with GroupNorm

## Observation

Training mIoU improves steadily, while validation fluctuates: 0.4949 at epoch 10, 0.2669 at epoch 13, and 0.5155 at epoch 16. Since the validation views and their points are fixed, evaluation sampling cannot explain these variations.

## Diagnostic

The epoch 16 and 22 checkpoints are evaluated before and after BatchNorm recalibration. Recalibration uses 20 training scans without gradients or weight updates. Evaluation uses the same 20 validation scans and two fixed views of 15,000 points.

| Checkpoint | Condition | Accuracy | Mean F1 | mIoU |
| --- | --- | ---: | ---: | ---: |
| Epoch 16 | Without recalibration | 0.8775 | 0.7546 | 0.6397 |
| Epoch 16 | With recalibration | 0.9121 | 0.8134 | 0.7222 |
| Epoch 22 | Without recalibration | 0.8317 | 0.7017 | 0.5916 |
| Epoch 22 | With recalibration | 0.9149 | 0.8312 | 0.7499 |

After recalibration, epoch 22 outperforms epoch 16. BatchNorm running statistics therefore mask part of the improvement in the model weights.

## Decision

All eight BatchNorm layers are replaced with `GroupNorm(8)`. GroupNorm does not maintain running statistics and is independent of batch history.

Only one GroupNorm run is required, with the existing BatchNorm run serving as the reference. All other parameters remain unchanged.
