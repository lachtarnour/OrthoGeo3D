import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.check_dataset_integrity import check_sample
from scripts.create_patient_splits import build_patient_random_split
from scripts.train_segmentation import build_early_stopping, build_scheduler
from src.augmentations import (
    RandomJitter,
    RandomRotation3D,
    RandomScale,
    RandomTranslation,
    build_train_augmentation,
)
from src.datasets.labels import (
    ARCH_CLASS_TO_FDI,
    CLASS_TO_FDI,
    FDI_TO_ARCH_CLASS,
    FDI_TO_CLASS,
    UPPER_ARCH_CLASS_TO_FDI,
    map_fdi_to_class,
)
from src.datasets.teeth3ds_processed import create_segmentation_dataloader
from src.datasets.teeth3ds_raw import build_processed_sample
from src.models.dgcnn import DGCNNSegmentation
from src.training.sampling import build_sampling_preprocessors, eval_view_ids
from src.training.checkpointing import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.loggers import WandbLogger
from src.training.checkpointing import checkpoint_configs
from src.training.tasks import SegmentationTask, Task
from src.training.trainer import Trainer
from src.utils.io import load_processed_sample, save_processed_sample
from src.utils.config import load_config


class TestSegmentationPipeline(unittest.TestCase):
    def test_data_sampling_loss_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = _write_fixture(Path(tmpdir))
            with patch.dict("os.environ", {"DATA_DIR": tmpdir}, clear=False):
                loader = create_segmentation_dataloader(config_path, split="train")
                batch = next(iter(loader))

        sampling_config = {
            "source_points": 128,
            "num_points": 64,
            "eval_views": 2,
        }
        augmentation = build_train_augmentation(
            _augmentation_config(),
            seed=42,
            feature_keys=("pos", "normal"),
        )
        train_preprocessor, eval_preprocessor = build_sampling_preprocessors(
            sampling_config,
            seed=42,
            train_transform=augmentation,
        )
        train_batch = train_preprocessor(batch, epoch=1)
        next_epoch_batch = train_preprocessor(batch, epoch=2)
        view_ids = eval_view_ids(sampling_config)
        val_view_zero = eval_preprocessor(batch, epoch=1, view_id=view_ids[0])
        val_view_zero_repeat = eval_preprocessor(batch, epoch=8, view_id=view_ids[0])
        val_view_one = eval_preprocessor(batch, epoch=1, view_id=view_ids[1])

        model = DGCNNSegmentation(
            input_channels=6,
            num_classes=17,
            k=4,
            emb_dims=64,
            dropout=0.0,
        )
        task = SegmentationTask(num_classes=17, loss_config=_loss_config())
        output = task.training_step(model, train_batch)

        self.assertEqual(batch["x"].shape, (2, 128, 6))
        self.assertEqual(train_batch["x"].shape, (2, 64, 6))
        self.assertTrue(torch.equal(train_batch["x"][..., 0:3], train_batch["pos"]))
        self.assertFalse(torch.equal(train_batch["y"], next_epoch_batch["y"]))
        self.assertEqual(view_ids, [0, 1])
        self.assertTrue(torch.equal(val_view_zero["y"], val_view_zero_repeat["y"]))
        self.assertFalse(torch.equal(val_view_zero["y"], val_view_one["y"]))
        self.assertEqual(output["logits"].shape, (2, 64, 17))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(output["confusion_matrix"].shape, (17, 17))
        for metric in (
            "accuracy",
            "miou",
            "loss_cross_entropy",
            "loss_dice",
            "loss_binary",
        ):
            self.assertIn(metric, output["metrics"])

    def test_point_cloud_augmentation_is_reproducible(self) -> None:
        torch.manual_seed(42)
        pos = torch.randn(2, 32, 3)
        normal = torch.nn.functional.normalize(torch.randn(2, 32, 3), dim=-1)
        target = torch.randint(0, 17, (2, 32))
        batch = {
            "scan_id": ["scan_a", "scan_b"],
            "pos": pos,
            "normal": normal,
            "x": torch.cat((pos, normal), dim=-1),
            "y": target,
        }
        augmentation = build_train_augmentation(
            _augmentation_config(),
            seed=42,
            feature_keys=("pos", "normal"),
        )
        self.assertIsNotNone(augmentation)

        first = augmentation(batch, epoch=3)
        repeated = augmentation(batch, epoch=3)
        next_epoch = augmentation(batch, epoch=4)

        self.assertTrue(torch.equal(first["pos"], repeated["pos"]))
        self.assertFalse(torch.equal(first["pos"], next_epoch["pos"]))
        self.assertFalse(torch.equal(first["pos"], pos))
        self.assertTrue(torch.equal(batch["pos"], pos))
        self.assertTrue(torch.equal(first["y"], target))
        self.assertTrue(torch.equal(first["x"][..., :3], first["pos"]))
        self.assertTrue(torch.equal(first["x"][..., 3:], first["normal"]))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(first["normal"], dim=-1),
                torch.ones(2, 32),
                atol=1.0e-6,
            )
        )

    def test_point_cloud_augmentation_preserves_geometry_contracts(self) -> None:
        torch.manual_seed(7)
        pos = torch.randn(128, 3)
        normal = torch.nn.functional.normalize(torch.randn(128, 3), dim=-1)
        point_cloud = {"pos": pos, "normal": normal}

        rotation = RandomRotation3D((5.0, 5.0, 10.0))(
            point_cloud,
            generator=torch.Generator().manual_seed(1),
        )
        original_edges = torch.linalg.vector_norm(pos[1:] - pos[:-1], dim=-1)
        rotated_edges = torch.linalg.vector_norm(
            rotation["pos"][1:] - rotation["pos"][:-1], dim=-1
        )
        self.assertTrue(torch.allclose(original_edges, rotated_edges, atol=1.0e-6))

        scaled = RandomScale((0.95, 1.05))(
            point_cloud,
            generator=torch.Generator().manual_seed(2),
        )
        scale = torch.linalg.vector_norm(
            scaled["pos"], dim=-1
        ) / torch.linalg.vector_norm(pos, dim=-1)
        self.assertGreaterEqual(float(scale.min()), 0.95 - 1.0e-6)
        self.assertLessEqual(float(scale.max()), 1.05 + 1.0e-6)
        self.assertLess(float(scale.max() - scale.min()), 1.0e-6)

        translated = RandomTranslation(0.01)(
            point_cloud,
            generator=torch.Generator().manual_seed(3),
        )
        offset = translated["pos"] - pos
        self.assertLessEqual(float(offset.abs().max()), 0.01 + 1.0e-7)
        self.assertLess(float((offset - offset[:1]).abs().max()), 2.0e-7)

        jittered = RandomJitter(0.002, 0.01)(
            point_cloud,
            generator=torch.Generator().manual_seed(4),
        )
        self.assertLessEqual(
            float((jittered["pos"] - pos).abs().max()),
            0.01 + 1.0e-7,
        )

    def test_processed_sample_preserves_shared_project_contract(self) -> None:
        rng = np.random.default_rng(42)
        vertices = rng.normal(size=(64, 3)).astype(np.float32)
        normals = rng.normal(size=(64, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        upper_fdi = np.asarray(list(UPPER_ARCH_CLASS_TO_FDI.values()))
        labels = np.resize(upper_fdi, len(vertices)).astype(np.int64)
        raw_scan = {
            "scan_id": "contract_upper",
            "patient_id": "contract",
            "jaw": "upper",
            "vertices": vertices,
            "normals": normals,
            "normal_source": "obj",
            "labels": labels,
            "instances": labels.copy(),
            "landmarks": [],
        }

        sample = build_processed_sample(raw_scan, num_points=32, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.pt"
            save_processed_sample(sample, path)
            loaded = load_processed_sample(path)

        errors, _ = check_sample(loaded, path.name, expected_num_points=32)
        self.assertEqual(errors, [])
        for key in (
            "pos_raw",
            "y_binary",
            "y_fdi",
            "y_fdi_class",
            "y_arch_class",
            "y_instance",
            "fdi_to_class",
            "class_to_fdi",
            "source_indices",
            "landmarks_raw",
            "tooth_centers_raw",
        ):
            self.assertIn(key, loaded)

    def test_segmentation_task_preserves_future_model_outputs(self) -> None:
        model = DictOutputModel()
        batch = {
            "x": torch.randn(2, 32, 6),
            "y": torch.randint(0, 17, (2, 32)),
        }
        task = SegmentationTask(num_classes=17, loss_config=_loss_config())

        output = task.training_step(model, batch)

        self.assertEqual(output["logits"].shape, (2, 32, 17))
        self.assertEqual(output["landmark_heatmaps"].shape, (2, 32, 6))

    def test_patient_random_split_keeps_patients_disjoint(self) -> None:
        records = [
            SimpleNamespace(
                scan_id=f"patient_{patient}_{jaw}",
                patient_id=f"patient_{patient}",
                jaw=jaw,
                landmark_path=None,
            )
            for patient in range(10)
            for jaw in ("upper", "lower")
        ]

        splits = build_patient_random_split(records, train_ratio=0.8, seed=42)
        train_patients = {record.patient_id for record in splits["train"]}
        val_patients = {record.patient_id for record in splits["val"]}

        self.assertFalse(train_patients & val_patients)
        self.assertEqual(
            train_patients | val_patients, {f"patient_{i}" for i in range(10)}
        )

    def test_evaluation_requires_checkpoint_configuration(self) -> None:
        train_config = {
            "run": {"seed": 7},
            "model": {},
            "sampling": {},
            "loss": {},
        }
        data_config = {"dataset": {"split_source": "teethseg22"}}
        checkpoint = {
            "config": {
                "train_config": train_config,
                "data_config": data_config,
            }
        }

        loaded_train, loaded_data = checkpoint_configs(checkpoint)

        self.assertEqual(loaded_train, train_config)
        self.assertEqual(loaded_data, data_config)
        with self.assertRaises(ValueError):
            checkpoint_configs({"config": train_config})

    def test_shared_trainer_accepts_a_non_segmentation_task(self) -> None:
        model = torch.nn.Linear(2, 1)
        task = RegressionTask()
        trainer = Trainer(
            model=model,
            task=task,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device="cpu",
            max_epochs=1,
        )
        batch = {
            "x": torch.ones(4, 2),
            "target": torch.zeros(4, 1),
        }

        trainer.fit([batch])

        self.assertEqual(trainer.global_step, 1)

    def test_training_controls_share_the_validation_monitor(self) -> None:
        config = load_config("configs/train/dgcnn_segmentation.yaml")
        training_config = config["training"]
        monitor = training_config["monitor"]
        mode = training_config["monitor_mode"]
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=training_config["lr"])

        scheduler, scheduler_monitor = build_scheduler(
            optimizer,
            scheduler_config=config["scheduler"],
            monitor=monitor,
            mode=mode,
        )
        early_stopping = build_early_stopping(
            config["early_stopping"],
            monitor=monitor,
            mode=mode,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler_monitor, monitor)
        self.assertEqual(early_stopping.monitor, monitor)
        self.assertEqual(early_stopping.mode, mode)

    def test_wandb_logger_writes_to_the_active_run(self) -> None:
        run = Mock()
        wandb = SimpleNamespace(
            init=Mock(return_value=run),
            Settings=lambda **kwargs: kwargs,
        )
        with patch.dict("sys.modules", {"wandb": wandb}):
            logger = WandbLogger(
                project="OrthoTwin3D",
                name="test-run",
                config={"seed": 42},
            )

        logger.log(
            {"train_loss": 1.25, "train_miou": 0.5},
            step=10,
            epoch=2,
            split="train",
        )
        logger.close()

        run.log.assert_called_once_with(
            {
                "epoch": 2,
                "global_step": 10,
                "train/loss": 1.25,
                "train/miou": 0.5,
            }
        )
        run.finish.assert_called_once_with()

    def test_early_stopping_is_applied_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            early_stopping = EarlyStopping(monitor="val_score", mode="max", patience=2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=0
            )
            trainer = Trainer(
                model=model,
                task=ValidationSequenceTask([0.5, 0.4, 0.3]),
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_step_metric="val_score",
                device="cpu",
                max_epochs=10,
                early_stopping=early_stopping,
                checkpoint_manager=CheckpointManager(
                    tmpdir, monitor="val_score", mode="max"
                ),
            )
            batch = {"x": torch.ones(1, 2)}

            trainer.fit([batch], [batch])

            self.assertEqual(trainer.global_step, 3)
            self.assertEqual(early_stopping.best_epoch, 1)
            self.assertEqual(early_stopping.stopped_epoch, 3)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)

            resumed_early_stopping = EarlyStopping(
                monitor="val_score", mode="max", patience=2
            )
            resumed_model = torch.nn.Linear(2, 1)
            resumed_trainer = Trainer(
                model=resumed_model,
                task=ValidationSequenceTask([0.2]),
                optimizer=torch.optim.SGD(resumed_model.parameters(), lr=0.1),
                device="cpu",
                early_stopping=resumed_early_stopping,
            )
            resumed_trainer.resume_from_checkpoint(str(Path(tmpdir) / "last.pt"))

            self.assertEqual(resumed_trainer.start_epoch, 4)
            self.assertEqual(resumed_early_stopping.best_metric, 0.5)
            self.assertEqual(resumed_early_stopping.bad_validation_count, 2)


class DictOutputModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(6, 17)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.projection(x)
        return {
            "logits": logits,
            "landmark_heatmaps": logits[..., :6],
        }


class RegressionTask(Task):
    def training_step(self, model: torch.nn.Module, batch: dict) -> dict[str, object]:
        prediction = model(batch["x"])
        loss = torch.nn.functional.mse_loss(prediction, batch["target"])
        return {
            "loss": loss,
            "metrics": {"loss": loss.detach()},
            "num_points": batch["x"].shape[0],
        }

    def validation_step(self, model: torch.nn.Module, batch: dict) -> dict[str, object]:
        return self.training_step(model, batch)


class ValidationSequenceTask(Task):
    def __init__(self, scores: list[float]) -> None:
        self.scores = iter(scores)

    def training_step(self, model: torch.nn.Module, batch: dict) -> dict[str, object]:
        loss = model(batch["x"]).sum() * 0.0
        return {"loss": loss, "metrics": {"loss": loss.detach()}}

    def validation_step(self, model: torch.nn.Module, batch: dict) -> dict[str, object]:
        score = next(self.scores)
        return {"metrics": {"score": score}, "num_points": 1}


def _write_fixture(root: Path) -> Path:
    processed_dir = root / "processed" / "teethseg22" / "train"
    processed_dir.mkdir(parents=True)
    torch.manual_seed(42)
    for sample_index in range(2):
        arch_labels = torch.arange(128, dtype=torch.long) % 17
        fdi_labels = torch.tensor(
            [UPPER_ARCH_CLASS_TO_FDI[int(value)] for value in arch_labels],
            dtype=torch.long,
        )
        fdi_classes = torch.as_tensor(
            map_fdi_to_class(fdi_labels.numpy()), dtype=torch.long
        )
        sample = {
            "scan_id": f"scan_{sample_index}",
            "patient_id": f"patient_{sample_index}",
            "jaw": "upper",
            "pos": torch.randn(128, 3),
            "normal": torch.randn(128, 3),
            "normal_source": "obj",
            "y_binary": (fdi_labels > 0).long(),
            "y_fdi": fdi_labels,
            "y_fdi_class": fdi_classes,
            "y_arch_class": arch_labels,
            "y_instance": fdi_labels,
            "fdi_to_class": FDI_TO_CLASS,
            "class_to_fdi": CLASS_TO_FDI,
            "fdi_to_arch_class": FDI_TO_ARCH_CLASS,
            "arch_class_to_fdi": ARCH_CLASS_TO_FDI["upper"],
        }
        save_processed_sample(sample, processed_dir / f"scan_{sample_index}.pt")

    config_path = root / "data.yaml"
    config_path.write_text(
        """
paths:
  processed_dir: null
dataset:
  split_source: teethseg22
dataloader:
  train:
    batch_size: 2
    num_workers: 0
    shuffle: false
  eval:
    batch_size: 2
    num_workers: 0
    shuffle: false
""",
        encoding="utf-8",
    )
    return config_path


def _loss_config() -> dict:
    return {
        "cross_entropy_weight": 1.0,
        "dice_weight": 1.0,
        "binary_weight": 0.5,
        "max_cross_entropy": 6.0,
        "max_binary": 6.0,
    }


def _augmentation_config() -> dict:
    return {
        "rotation_degrees": [5.0, 5.0, 10.0],
        "scale_range": [0.95, 1.05],
        "translation_range": 0.01,
        "jitter_std": 0.001,
        "jitter_clip": 0.003,
    }


if __name__ == "__main__":
    unittest.main()
