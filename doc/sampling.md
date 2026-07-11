# Sampling Protocol

Sampling is split into two decisions:

- preprocessing sampling: build a stable scan-level representation from each mesh;
- training sampling: build model-size views from the processed scan at runtime.

This separation keeps the processed dataset reproducible while allowing training
views to vary under controlled rules.

## Context

Teeth3DS/Teeth3DS+ scans are dense intraoral meshes. The public benchmark
contains 1,800 scans from 900 patients and supports tooth segmentation,
labeling, and landmark tasks. The current pipeline uses a higher-resolution
60,000-point processed representation before runtime view sampling.

Naive downsampling is risky for dental scans. Most points belong to gingiva or
large tooth surfaces, while small teeth, contacts, and tooth-gingiva boundaries
carry much of the segmentation signal.

## Sampling Options

| Strategy | Status | Notes |
| --- | --- | --- |
| Full mesh | Not used for DGCNN | Best geometric fidelity, but high memory and unstable batch cost. |
| Farthest Point Sampling | Used | Standard geometry-only coverage sampler. |
| Fixed 60k preprocessing | Used | Keeps roughly half of the median raw mesh while staying compact enough for storage. |
| Poisson Disk sampling | Considered | Used in dental segmentation repos to enforce spatial spacing. It may create new vertices depending on implementation. |
| Poisson Disk simplification | Considered | Keeps existing vertices while enforcing spacing. More mesh-specific than the current pipeline. |
| Boundary-aware sampling | Not selected | It can help training, but it uses supervision to define the processed dataset. |
| Class-aware sampling | Not selected | It can preserve rare teeth, but it biases validation/test point selection. |

## Selected Preprocessing Sampler

The selected preprocessing method is `fps`.

It applies one rule:

1. Select 60,000 existing mesh vertices with farthest point sampling using only
   normalized geometry.

Current setting:

```text
input mesh -> 60,000 processed points
```

The goal is to keep the processed train/validation/test scans neutral. The
preprocessing step should not use labels to decide which points are evaluated.
Rare teeth and tooth/background imbalance are handled by the loss and training
protocol instead.

## Training-Time Sampling

The processed `.pt` files remain at 60,000 points. DGCNN receives smaller
runtime views because GPU memory scales with batch size, point count, and
neighbor graph construction.

Current GPU setting:

```text
processed scan size = 60,000 points
model view size     = 15,000 points
stable core         = 10,000 points
changing subset     = 5,000 points
views per scan      = 10
```

This is deterministic overlapping multi-view sampling:

- each scan gets a stable point order from `seed + scan_id`;
- each view shares the same core points;
- the remaining points rotate across views;
- training changes view with the epoch;
- validation runs all views and aggregates the metrics.

This gives an approximate 67% stable / 33% changing protocol. Ten views cover
the full 60,000-point processed scan once: 10,000 shared core points plus
10 rotating blocks of 5,000 points. Validation is heavier, so the maintained
configs run it every 5 epochs and on the final epoch.

## Current Choice

The current protocol is:

```text
preprocessing: geometry-only FPS, 60,000 points
training:      deterministic overlapping multi-view, 15,000 points
validation:    all 10 deterministic views, every 5 epochs
selection:     best checkpoint by val_miou
```

This choice avoids label-aware validation/test sampling, reduces DGCNN memory
pressure, supports larger physical batches, and makes validation less sensitive
to a single sampled neighborhood graph.

## References

- Teeth3DS+ benchmark: https://crns-smartvision.github.io/teeth3ds/
- Teeth3DS+ paper: https://arxiv.org/html/2210.06094v3
- PyTorch Geometric Teeth3DS dataset: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Teeth3DS.html
- ToothGroupNetwork repository: https://github.com/limhoyeon/ToothGroupNetwork
- Rotation-Invariant Tooth Seg repository: https://github.com/Namkwangwoon/Rotation-Invariant-Tooth-Seg/
