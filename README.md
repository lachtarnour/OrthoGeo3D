# OrthoTwin3D

DGCNN pipeline for 3D intra-oral tooth segmentation on Teeth3DS.

![OrthoTwin3D segmentation examples](doc/assets/orthotwin3d_segmentation_examples.png)

## Project Plan

OrthoTwin3D aims to convert a 3D intra-oral scan into structured anatomical
outputs: point-wise tooth labels, tooth instances, landmarks and interpretable
geometric measurements.

```text
scan -> segmentation -> tooth instances -> landmarks -> geometry
```

The data foundation and PyTorch input pipeline are implemented. Current work
focuses on finalizing the DGCNN segmentation baseline before introducing the
following components.

1. **DGCNN segmentation:** establish a reproducible point-wise FDI
   segmentation baseline.
2. **Instance postprocessing:** convert predicted labels into clean tooth
   instances and centers.
3. **Landmark and geometry baselines:** predict landmarks from tooth crops and
   compute non-differentiable geometric measurements.
4. **Multi-task DGCNN:** learn segmentation and landmarks with a shared
   backbone.
5. **OrthoTwin3D-GC:** add differentiable center, width, axis and inter-tooth
   consistency constraints.
6. **PTv3 fine-tuning:** repeat the segmentation, multi-task and
   geometry-constrained comparison with a pretrained transformer backbone.
7. **Evaluation and reporting:** run ablations, robustness tests and final
   DGCNN-versus-PTv3 comparisons.

The central research comparison is segmentation-only versus multi-task versus
geometry-constrained learning. Derived geometric measurements are research
outputs and are not intended as clinical diagnoses or treatment plans.

## Baseline

The maintained baseline is a DGCNN trained for point-wise segmentation on the
patient-disjoint Teeth3DS split.

| Component | Setting |
| --- | --- |
| Data | 1,196 training scans and 592 validation scans |
| Input | 15,000 points sampled from 60,000 preprocessed vertices; normalized position and normals |
| Train augmentation | 3D rotation up to 5 degrees on X/Y and 10 degrees on Z, 0.95-1.05 scaling, +/-0.01 translation and 0.001 Gaussian jitter clipped at 0.003 |
| Target | 17 classes: background and 16 jaw-normalized tooth positions |
| Model | DGCNN, `k=20`, embedding dimension 1,024, `GroupNorm(8)`, dropout 0.5 |
| Optimization | AdamW, batch size 16, learning rate `1e-3`, weight decay `1e-4`, mixed precision |
| Objective | Cross-entropy + Dice + 0.5 binary tooth/background loss |
| Evaluation | Two fixed validation views; best checkpoint selected by validation mIoU |
| Training control | ReduceLROnPlateau (`factor=0.8`, `patience=2`) and early stopping (`patience=7`, `min_delta=0.001`) |

### Results

Results are measured on the validation split with the two fixed evaluation
views.

| Model | Best epoch | Validation loss | Accuracy | Mean F1 | mIoU | Learning curve | Remarks |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| DGCNN, GroupNorm, no augmentation | 57 | 0.7930 | 0.8898 | 0.7310 | 0.6264 | [View curves](doc/assets/learning_curves/dgcnn_groupnorm_no_augmentation.png) | Fixed-evaluation train-val mIoU gap: 0.1858. Train-only geometric augmentation proposed. |
| DGCNN, GroupNorm, train augmentation | 36 | 0.6649 | 0.9067 | 0.7593 | 0.6682 | [View curves](doc/assets/learning_curves/dgcnn_groupnorm_with_augmentation.png) | Gap reduced to 0.1078. Rare third-molar classes 1 and 16 occur in only 9 train scans, versus 79 and 73 validation scans. |

## Data

`DATA_DIR` contains persistent data. `OUTPUT_DIR` contains checkpoints and logs.
Both default to local project folders and are set to `/workspace/data` and
`/workspace/outputs` in Docker.

```text
DATA_DIR/
  raw/
  splits/teethseg22/
  processed/teethseg22/
    train/*.pt
    val/*.pt
```

The patient-disjoint protocol, source lists and integrity reports are described
in [Data splitting](doc/data_splitting.md).

Prepare the dataset:

```bash
python -m scripts.download_teeth3ds
python -m scripts.create_patient_splits
python -m scripts.prepare_data --num_points 60000 --num_workers 4
python -m scripts.check_dataset_integrity --allow_skipped
```

## Training

```bash
python -m scripts.train_segmentation
```

Resume from a checkpoint:

```bash
python -m scripts.train_segmentation \
  --resume "$OUTPUT_DIR/teethseg22_dgcnn_groupnorm/checkpoints/last.pt"
```

Training keeps the highest `val_miou` in `best.pt` and stops after seven
consecutive validations without an improvement of at least `0.001`.

Evaluate a checkpoint:

```bash
python -m scripts.evaluate_segmentation \
  --checkpoint "$OUTPUT_DIR/teethseg22_dgcnn_groupnorm/checkpoints/best.pt"
```

The only maintained training configuration is
[`configs/train/dgcnn_segmentation.yaml`](configs/train/dgcnn_segmentation.yaml).
