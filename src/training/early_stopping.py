from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class EarlyStopping:
    monitor: str
    mode: Literal["min", "max"]
    patience: int
    min_delta: float = 0.0
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_validation_count: int = 0
    stopped_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")

    def update(self, metrics: dict[str, float], epoch: int) -> bool:
        if self.monitor not in metrics:
            raise KeyError(
                f"Early-stopping metric {self.monitor!r} is missing at epoch {epoch}"
            )

        value = float(metrics[self.monitor])
        if self._is_improvement(value):
            self.best_metric = value
            self.best_epoch = int(epoch)
            self.bad_validation_count = 0
            return False

        self.bad_validation_count += 1
        if self.bad_validation_count < self.patience:
            return False

        self.stopped_epoch = int(epoch)
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_validation_count": self.bad_validation_count,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("monitor", "mode", "patience", "min_delta"):
            if state.get(key) != getattr(self, key):
                raise ValueError(
                    f"Early-stopping checkpoint mismatch for {key}: "
                    f"{state.get(key)!r} != {getattr(self, key)!r}"
                )

        self.best_metric = _optional_float(state.get("best_metric"))
        self.best_epoch = _optional_int(state.get("best_epoch"))
        self.bad_validation_count = int(state.get("bad_validation_count", 0))
        self.stopped_epoch = _optional_int(state.get("stopped_epoch"))

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return value < self.best_metric - self.min_delta
        return value > self.best_metric + self.min_delta


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
