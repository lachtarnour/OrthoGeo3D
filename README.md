# OrthoGeo3D

OrthoGeo3D is a research pipeline for analyzing 3D intra-oral scans from
Teeth3DS / Teeth3DS+ and 3DTeethLand.

The goal is to turn a 3D scan into a structured anatomical representation:

- tooth / gingiva segmentation and per-point FDI labels;
- tooth instances and tooth centers;
- anatomical landmarks;
- orthodontic-inspired geometric measurements;
- multi-task models and differentiable geometric constraints.

This project is not intended for clinical diagnosis. The geometric measurements
are research indices derived from public annotations and must be interpreted as
anatomical consistency signals.

![OrthoGeo3D segmentation examples](doc/assets/orthotwin3d_segmentation_examples.png)

## Research Plan

The scientific program compares increasingly structured models:

```text
DGCNN segmentation only
vs
DGCNN multi-task segmentation + landmarks
vs
OrthoGeo3D-GC with geometric constraints
vs
PTv3 fine-tuned segmentation
vs
PTv3 fine-tuned multi-task
vs
PTv3-GC fine-tuned
```

The pipeline follows this order:

1. build a reliable scan-level dataset;
2. train a DGCNN FDI segmentation baseline;
3. convert FDI predictions into tooth instances;
4. generate tooth crops for landmark detection;
5. compute geometric indices;
6. add multi-task learning, then geometric constraints;
7. run ablations, robustness analyses, and backbone comparisons.

The full scientific plan is documented in
`docs/PLAN_COMPLET_ORTHOTWIN3D_GC.md`. Step-by-step notes are in `docs/steps/`.

## Current Status

Already implemented:

- Data foundation: raw Teeth3DS / 3DTeethLand loading, preprocessing, FDI
  remapping, landmark association, tooth centers, and `.pt` export.
- Dataset reliability: patient-level splits, processed dataset integrity checks,
  and a configurable PyTorch segmentation dataloader.
- Baseline model: DGCNN semantic segmentation for per-point FDI prediction.
- Training foundation: generic `Trainer` + `Task` architecture, segmentation
  task, metrics, logging, checkpoints, and resume support.
- Validation: unit tests for the dataloader, DGCNN forward pass, training task,
  and checkpointing.

Processed sample format:

```python
sample = {
    "scan_id": str,
    "patient_id": str,
    "jaw": str,
    "pos_raw": Tensor[N, 3],
    "pos": Tensor[N, 3],
    "normal": Tensor[N, 3],
    "curvature": Tensor[N] | None,
    "y_binary": Tensor[N],
    "y_fdi": Tensor[N],
    "y_fdi_class": Tensor[N],
    "y_instance": Tensor[N],
    "landmarks_raw": dict,
    "landmarks_norm": dict,
    "landmark_to_tooth": list,
    "tooth_centers_raw": dict,
    "tooth_centers_norm": dict,
    "center": Tensor[3],
    "scale": float,
}
```

`y_fdi` keeps the real FDI labels. `y_fdi_class` is used by
`CrossEntropyLoss`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional: define an external data directory in `.env`.

```bash
DATA_DIR=/path/to/data
```

Without `.env`, the project uses `data/`.

## Usage

Download or verify the data:

```bash
python scripts/download_all_teeth3ds.py
```

Create patient-level splits:

```bash
python scripts/create_patient_splits.py --source patient_random
```

Preprocess all three splits:

```bash
python scripts/prepare_data.py --all_splits --num_workers 4 --sampling fps
```

Check the processed dataset:

```bash
python scripts/check_dataset_integrity.py
```

Quickly test the segmentation dataloader:

```bash
python - <<'PY'
from src.datasets import create_segmentation_dataloader

loader = create_segmentation_dataloader("configs/data.yaml", split="train", limit=2)
batch = next(iter(loader))
print(batch["x"].shape)  # [B, 30000, 6]
print(batch["y"].shape)  # [B, 30000]
PY
```

Run the tests:

```bash
python -m unittest discover -s tests
```

Run the DGCNN segmentation training:

```bash
python scripts/train_segmentation.py --config configs/train/dgcnn_segmentation.yaml
```

The training config uses all train/val scans and randomly samples 4096 points
per scan inside the training loop from the 30000-point dataloader batch. The
processed dataset and dataloader keep the full 30000-point contract; the trainer
samples the model batch.

Run outputs are written to:

```text
outputs/experiments/<experiment_name>/
```

with:

```text
config.yaml
metrics.jsonl
checkpoints/last.pt
checkpoints/best.pt
```

Resume from a checkpoint:

```bash
python scripts/train_segmentation.py \
  --config configs/train/dgcnn_segmentation.yaml \
  --resume outputs/experiments/dgcnn_random4096/checkpoints/last.pt
```

## Useful Structure

```text
configs/data.yaml                     # dataset and dataloaders
configs/train/dgcnn_segmentation.yaml # initial training config
scripts/prepare_data.py               # raw -> .pt
scripts/check_dataset_integrity.py     # processed dataset validation
scripts/train_segmentation.py          # segmentation training entrypoint
src/datasets/                          # raw and processed datasets
src/models/dgcnn.py                    # DGCNN segmentation
src/training/                          # generic trainer, tasks, metrics, logs
docs/                                  # scientific plan and step notes
```
