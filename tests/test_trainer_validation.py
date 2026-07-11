import unittest

import torch

from src.training.trainer import Trainer


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class TinyTask:
    def __init__(self) -> None:
        self.validation_view_ids: list[int] = []

    def training_step(self, model: torch.nn.Module, batch: dict) -> dict:
        loss = model.weight * 0.0 + 1.0
        return {"loss": loss, "metrics": {"loss": loss.detach()}}

    def validation_step(self, model: torch.nn.Module, batch: dict) -> dict:
        view_id = int(batch.get("view_id", -1))
        self.validation_view_ids.append(view_id)
        loss = torch.tensor(float(view_id + 1))
        return {"loss": loss, "metrics": {"loss": loss}}


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, str, dict[str, float]]] = []

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        self.records.append((epoch, split, metrics))

    def close(self) -> None:
        return None


class TestTrainerValidation(unittest.TestCase):
    def test_validation_can_run_every_n_epochs(self) -> None:
        task = TinyTask()
        logger = RecordingLogger()
        model = TinyModel()
        trainer = Trainer(
            model=model,
            task=task,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device="cpu",
            logger=logger,
            max_epochs=3,
            validation_every_epochs=2,
            validate_last_epoch=False,
        )

        trainer.fit([{"x": torch.ones(1, 1)}], [{"x": torch.ones(1, 1)}])

        val_epochs = [epoch for epoch, split, _ in logger.records if split == "val"]
        self.assertEqual(val_epochs, [2])
        train_records = [metrics for _, split, metrics in logger.records if split == "train"]
        self.assertEqual(train_records[0]["train_lr"], 0.1)

    def test_epoch_logging_can_run_every_n_epochs(self) -> None:
        task = TinyTask()
        logger = RecordingLogger()
        model = TinyModel()
        trainer = Trainer(
            model=model,
            task=task,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device="cpu",
            logger=logger,
            max_epochs=7,
            validation_every_epochs=3,
            validate_last_epoch=True,
            log_every_epochs=3,
        )

        trainer.fit([{"x": torch.ones(1, 1)}], [{"x": torch.ones(1, 1)}])

        train_epochs = [epoch for epoch, split, _ in logger.records if split == "train"]
        val_epochs = [epoch for epoch, split, _ in logger.records if split == "val"]
        self.assertEqual(train_epochs, [3, 6, 7])
        self.assertEqual(val_epochs, [3, 6, 7])

    def test_validation_aggregates_all_eval_views(self) -> None:
        task = TinyTask()
        model = TinyModel()

        def eval_preprocessor(batch: dict, view_id: int | None = None, **_) -> dict:
            item = dict(batch)
            item["view_id"] = view_id
            return item

        trainer = Trainer(
            model=model,
            task=task,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device="cpu",
            eval_batch_preprocessor=eval_preprocessor,
            eval_view_ids=[0, 1, 2],
        )

        metrics = trainer.evaluate([{"x": torch.ones(1, 1)}], epoch=1)

        self.assertEqual(task.validation_view_ids, [0, 1, 2])
        self.assertEqual(metrics["eval_views"], 3.0)
        self.assertEqual(metrics["loss"], 2.0)

    def test_optimizer_steps_every_training_batch(self) -> None:
        task = TinyTask()
        model = TinyModel()
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        trainer = Trainer(
            model=model,
            task=task,
            optimizer=optimizer,
            device="cpu",
            max_epochs=1,
        )

        trainer.fit([{"x": torch.ones(1, 1)} for _ in range(5)])

        self.assertEqual(optimizer.step_count, 5)

    def test_scheduler_steps_once_per_epoch(self) -> None:
        task = TinyTask()
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer = Trainer(
            model=model,
            task=task,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cpu",
            max_epochs=2,
            validation_every_epochs=1,
        )

        trainer.fit([{"x": torch.ones(1, 1)}], [{"x": torch.ones(1, 1)}])

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.025)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


if __name__ == "__main__":
    unittest.main()
