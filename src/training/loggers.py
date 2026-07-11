import json
import time
from pathlib import Path
from typing import Any

from src.training.utils import ensure_dir
from src.utils.logger import get_logger


WANDB_METRIC_NAMES = ("loss", "accuracy", "miou", "mean_f1", "lr")
WANDB_SPLITS = {"train_step", "train", "val"}


class BaseLogger:
    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ConsoleLogger(BaseLogger):
    def __init__(self, log_every_steps: int = 1, log_train_steps: bool = True) -> None:
        self.log_every_steps = max(1, int(log_every_steps))
        self.log_train_steps = bool(log_train_steps)
        self.logger = get_logger("training")

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        if split == "train_step" and not self.log_train_steps:
            return
        if split == "train_step" and step % self.log_every_steps != 0:
            return
        display_metrics = {
            key: value for key, value in metrics.items() if _metric_name_for_split(key, split) in WANDB_METRIC_NAMES
        }
        metric_text = " | ".join(f"{key}={value:.4f}" for key, value in sorted(display_metrics.items()))
        self.logger.info("epoch=%s step=%s split=%s | %s", epoch, step, split, metric_text)


class JsonlLogger(BaseLogger):
    def __init__(self, path: str | Path, log_train_steps: bool = True) -> None:
        self.path = Path(path)
        self.log_train_steps = bool(log_train_steps)
        ensure_dir(self.path.parent)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        if split == "train_step" and not self.log_train_steps:
            return
        record: dict[str, Any] = {
            "time": time.time(),
            "step": int(step),
            "epoch": int(epoch),
            "split": split,
            "metrics": metrics,
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        name: str,
        config: dict[str, Any] | None = None,
        log_every_steps: int = 1,
        log_train_steps: bool = True,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb logging was enabled, but the wandb package is not installed") from exc

        self.wandb = wandb
        self.log_every_steps = max(1, int(log_every_steps))
        self.log_train_steps = bool(log_train_steps)
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            settings=wandb.Settings(x_disable_stats=True),
        )
        self._define_metrics()

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        if split == "train_step" and not self.log_train_steps:
            return
        if split == "train_step" and step % self.log_every_steps != 0:
            return
        payload = {
            "epoch": int(epoch),
            "global_step": int(step),
        }
        payload.update(_format_wandb_metrics(metrics, split=split))
        self.wandb.log(payload)

    def close(self) -> None:
        self.run.finish()

    def _define_metrics(self) -> None:
        self.run.define_metric("epoch")
        self.run.define_metric("global_step")
        for metric_name in WANDB_METRIC_NAMES:
            self.run.define_metric(f"train_step/{metric_name}", step_metric="global_step")
            self.run.define_metric(f"train/{metric_name}", step_metric="epoch")
            self.run.define_metric(f"val/{metric_name}", step_metric="epoch")
        self.run.define_metric("train/loss", summary="min")
        self.run.define_metric("val/loss", summary="min")
        self.run.define_metric("train/accuracy", summary="max")
        self.run.define_metric("val/accuracy", summary="max")
        self.run.define_metric("train/miou", summary="max")
        self.run.define_metric("val/miou", summary="max")
        self.run.define_metric("train/mean_f1", summary="max")
        self.run.define_metric("val/mean_f1", summary="max")
        self.run.define_metric("train/lr", summary="last")


class CompositeLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger] | None = None) -> None:
        self.loggers = loggers or []

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step, epoch=epoch, split=split)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def _format_wandb_metrics(metrics: dict[str, float], split: str) -> dict[str, float]:
    if split not in WANDB_SPLITS:
        return {}

    formatted = {}
    for key, value in metrics.items():
        metric_name = _metric_name_for_split(key, split)
        if metric_name in WANDB_METRIC_NAMES:
            formatted[f"{split}/{metric_name}"] = value
    return formatted


def _metric_name_for_split(key: str, split: str) -> str:
    if split == "train_step":
        return key.removeprefix("train_step_")
    return key.removeprefix(f"{split}_")
