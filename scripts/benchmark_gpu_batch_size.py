#!/usr/bin/env python
import argparse
import copy
import gc
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.teeth3ds_processed import Teeth3DSSegmentationDataset, create_segmentation_dataloader
from src.models import DGCNNSegmentation
from src.training import SegmentationTask
from src.training.sampling import build_sampling_preprocessors
from src.training.utils import get_device, move_to_device, set_seed
from src.utils.config import load_config


DEFAULT_BATCH_SIZES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    optim_config = config.get("optim", {})
    data_config = config.get("data", {})
    seed = int(config.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(args.device or optim_config.get("device", "cuda"))
    if device.type != "cuda":
        raise RuntimeError("This benchmark needs a CUDA GPU.")
    configure_cuda_performance(config.get("performance", {}), device=device)

    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * float(args.target_utilization)
    batch_sizes = parse_batch_sizes(args.batch_sizes, args.max_batch_size)

    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"total_vram={format_gib(total_memory)}")
    print(f"target={format_gib(target_memory)} ({args.target_utilization:.0%})")
    print(f"config={args.config}")
    print("batch_size,status,peak_allocated,peak_reserved,loss")

    recommended: int | None = None
    for batch_size in batch_sizes:
        result = benchmark_batch_size(config, batch_size=batch_size, device=device, args=args, seed=seed)
        status = "ok" if result["ok"] else "oom"
        print(
            f"{batch_size},{status},"
            f"{format_gib(result['peak_allocated'])},"
            f"{format_gib(result['peak_reserved'])},"
            f"{result.get('loss', '')}"
        )

        if result["ok"] and result["peak_allocated"] <= target_memory:
            recommended = batch_size
        if not result["ok"] and args.stop_on_oom:
            break

    if recommended is None:
        print("recommended_batch_size=none")
    else:
        print(f"recommended_batch_size={recommended}")


def benchmark_batch_size(
    config: dict[str, Any],
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loader = None
    batch = None
    model = None
    task = None
    optimizer = None
    scaler = None
    output = None
    loss = None

    try:
        data_config = copy.deepcopy(config.get("data", {}))
        dataloader_override = copy.deepcopy(data_config.get("dataloader", {}))
        train_loader_config = dataloader_override.setdefault("train", {})
        train_loader_config["batch_size"] = int(batch_size)
        train_loader_config["shuffle"] = False
        train_loader_config["num_workers"] = int(args.num_workers)
        train_loader_config["persistent_workers"] = False

        loader = create_segmentation_dataloader(
            data_config.get("config_path", "configs/data.yaml"),
            split=data_config.get("train_split", "train"),
            limit=max(int(batch_size), int(args.min_samples)),
            dataloader_config_override=dataloader_override,
        )
        batch = next(iter(loader))
        train_preprocessor, _ = build_sampling_preprocessors(data_config, seed=seed)
        if train_preprocessor is not None:
            batch = train_preprocessor(batch, epoch=args.epoch, split="train")
        batch = move_to_device(batch, device)

        model = build_model(config.get("model", {})).to(device)
        train_dataset = loader.dataset
        if not isinstance(train_dataset, Teeth3DSSegmentationDataset):
            raise TypeError("loader.dataset must be a Teeth3DSSegmentationDataset")
        num_classes = int(config.get("model", {}).get("num_classes", train_dataset.num_classes))
        task = SegmentationTask(num_classes=num_classes, loss_config=config.get("loss", {}))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("optim", {}).get("lr", 1.0e-3)),
            weight_decay=float(config.get("optim", {}).get("weight_decay", 1.0e-4)),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("optim", {}).get("amp", False)))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=bool(config.get("optim", {}).get("amp", False))):
            output = task.training_step(model, batch)
            loss = output["loss"]
        scaler.scale(loss).backward()
        if args.optimizer_step:
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize(device)

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        loss_value = float(loss.detach().cpu().item())
        ok = True
    except RuntimeError as exc:
        if not is_cuda_oom(exc):
            raise
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        loss_value = ""
        ok = False
    finally:
        del loader, batch, model, task, optimizer, scaler, output, loss
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "ok": ok,
        "peak_allocated": int(peak_allocated),
        "peak_reserved": int(peak_reserved),
        "loss": f"{loss_value:.4f}" if ok else "",
    }


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    return DGCNNSegmentation(
        input_channels=int(model_config.get("input_channels", 6)),
        num_classes=int(model_config.get("num_classes", 17)),
        k=int(model_config.get("k", 20)),
        emb_dims=int(model_config.get("emb_dims", 1024)),
        dropout=float(model_config.get("dropout", 0.5)),
    )


def configure_cuda_performance(performance_config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    use_tf32 = bool(performance_config.get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if use_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(performance_config.get("float32_matmul_precision", "high")))
    torch.backends.cudnn.benchmark = bool(performance_config.get("cudnn_benchmark", False))


def parse_batch_sizes(raw: str | None, max_batch_size: int) -> list[int]:
    if raw:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        values = [value for value in DEFAULT_BATCH_SIZES if value <= int(max_batch_size)]
        if int(max_batch_size) not in values:
            values.append(int(max_batch_size))
    return sorted(set(value for value in values if value > 0))


def is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def format_gib(value: float | int) -> str:
    return f"{float(value) / (1024 ** 3):.2f}GiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one training batch for GPU memory usage.")
    parser.add_argument("--config", default="configs/train/dgcnn_segmentation_ovh_gpu.yaml")
    parser.add_argument("--batch-sizes", help="Comma-separated batch sizes to test.")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--target-utilization", type=float, default=0.80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--optimizer-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-oom", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
