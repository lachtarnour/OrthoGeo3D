import json
import time
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir
from src.utils.logger import get_logger


METRIC_NAMES = ("loss", "accuracy", "miou", "mean_f1", "lr")


class BaseLogger:
    def log(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        split: str,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ConsoleLogger(BaseLogger):
    def __init__(self) -> None:
        self.logger = get_logger("training")

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        split: str,
    ) -> None:
        display = {
            key: value
            for key, value in metrics.items()
            if key.removeprefix(f"{split}_") in METRIC_NAMES
        }
        text = " | ".join(
            f"{key}={value:.4f}" for key, value in sorted(display.items())
        )
        self.logger.info("epoch=%s step=%s split=%s | %s", epoch, step, split, text)


class JsonlLogger(BaseLogger):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._file = self.path.open("a", encoding="utf-8")

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        split: str,
    ) -> None:
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
        config: dict[str, Any],
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but wandb is not installed"
            ) from exc

        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            settings=wandb.Settings(x_disable_stats=True),
        )
        self.run.define_metric("epoch")
        self.run.define_metric("global_step")
        for split in ("train", "val"):
            self.run.define_metric(f"{split}/*", step_metric="epoch")
            for metric in METRIC_NAMES:
                self.run.define_metric(f"{split}/{metric}", step_metric="epoch")

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        split: str,
    ) -> None:
        payload = {"epoch": int(epoch), "global_step": int(step)}
        for key, value in metrics.items():
            metric = key.removeprefix(f"{split}_")
            payload[f"{split}/{metric}"] = value
        self.run.log(payload)

    def close(self) -> None:
        self.run.finish()


class CompositeLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger] | None = None) -> None:
        self.loggers = loggers or []

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        split: str,
    ) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step, epoch=epoch, split=split)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
