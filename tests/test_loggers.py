import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.training.loggers import JsonlLogger, WandbLogger


class TestLoggers(unittest.TestCase):
    def test_jsonl_logger_can_skip_train_step_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = JsonlLogger(path, log_train_steps=False)

            logger.log({"loss": 1.0}, step=1, epoch=1, split="train_step")
            logger.log({"loss": 0.5}, step=10, epoch=1, split="train")
            logger.close()

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["split"], "train")
        self.assertEqual(records[0]["metrics"]["loss"], 0.5)

    def test_wandb_logger_uses_wandb_auto_step(self) -> None:
        logged = []

        class FakeRun:
            def define_metric(self, *args, **kwargs) -> None:
                return None

            def finish(self) -> None:
                return None

        fake_wandb = SimpleNamespace(
            Settings=lambda **kwargs: kwargs,
            init=lambda **kwargs: FakeRun(),
            log=lambda payload, **kwargs: logged.append((payload, kwargs)),
        )

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            logger = WandbLogger(project="test", name="run", log_train_steps=False)
            logger.log({"train_loss": 1.0}, step=450, epoch=3, split="train")
            logger.close()

        self.assertEqual(len(logged), 1)
        payload, kwargs = logged[0]
        self.assertEqual(kwargs, {})
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["global_step"], 450)
        self.assertEqual(payload["train/loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
