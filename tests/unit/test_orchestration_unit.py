from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn

from orchestration.handlers.base_train import run_base_train
from orchestration.handlers.build_oof import run_build_oof
from orchestration.handlers.optuna_tune import run_optuna_tune
from orchestration.handlers.train_meta import run_train_meta
from orchestration.handlers.train_stack import _extract_best_metric, run_train_stack
from orchestration.job_contracts import JobSpec
from orchestration.jobs import _ensure_run_dir, build_job_spec, execute_job
from orchestration.models import PairOrchestrator, _read_json_file, load_meta_head_state


class TestPairOrchestrator(unittest.TestCase):
    def test_unknown_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            PairOrchestrator(nn.Linear(8, 2), nn.Linear(8, 2), mode="unknown")

    def test_set_backbones_trainable_false_then_true(self) -> None:
        pair = PairOrchestrator(nn.Linear(8, 2), nn.Linear(8, 2), mode="sum")

        pair.set_backbones_trainable(False)
        self.assertTrue(all(not p.requires_grad for p in pair.model_a.parameters()))
        self.assertTrue(all(not p.requires_grad for p in pair.model_b.parameters()))

        pair.set_backbones_trainable(True)
        self.assertTrue(all(p.requires_grad for p in pair.model_a.parameters()))
        self.assertTrue(all(p.requires_grad for p in pair.model_b.parameters()))

    def test_load_meta_head_state_requires_meta_head(self) -> None:
        pair = PairOrchestrator(nn.Linear(8, 2), nn.Linear(8, 2), mode="sum")
        with tempfile.TemporaryDirectory() as td:
            weights_path = Path(td) / "meta.pt"
            weights_path.write_bytes(b"fake")
            with self.assertRaises(ValueError):
                load_meta_head_state(pair, str(weights_path))

    def test_load_meta_head_state_switches_mode_to_meta(self) -> None:
        pair = PairOrchestrator(nn.Linear(8, 2), nn.Linear(8, 2), mode="meta")
        expected_state = pair.meta_head.state_dict()
        with tempfile.TemporaryDirectory() as td:
            weights_path = Path(td) / "meta.pt"
            torch.save(expected_state, weights_path)
            reloaded = load_meta_head_state(pair, str(weights_path))
        self.assertIs(reloaded, pair)
        self.assertEqual(pair.mode, "meta")

    def test_read_json_file_raises_on_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "broken.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_json_file(path, label="test")

    def test_read_json_file_raises_on_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                _read_json_file(missing, label="test")

    def test_read_json_file_raises_on_non_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "list.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_json_file(path, label="test")


class TestTrainStackMetricExtraction(unittest.TestCase):
    def test_extract_best_metric_prefers_validation(self) -> None:
        history = {"val": {"contest_metric": [0.1, 0.55, 0.21]}}
        best_metric, best_epoch = _extract_best_metric(history)
        self.assertEqual(best_metric, 0.55)
        self.assertEqual(best_epoch, 2)

    def test_extract_best_metric_fallbacks_to_train(self) -> None:
        history = {"train": {"contest_metric": [0.2, 0.3, 0.25]}}
        best_metric, best_epoch = _extract_best_metric(history)
        self.assertEqual(best_metric, 0.3)
        self.assertEqual(best_epoch, 2)

    def test_extract_best_metric_handles_empty(self) -> None:
        best_metric, best_epoch = _extract_best_metric({})
        self.assertIsNone(best_metric)
        self.assertIsNone(best_epoch)

    def test_extract_best_metric_ignores_non_list_metrics(self) -> None:
        history = {"val": {"contest_metric": "not-a-list"}}
        best_metric, best_epoch = _extract_best_metric(history)
        self.assertIsNone(best_metric)
        self.assertIsNone(best_epoch)


class TestHandlers(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = JobSpec(
            job_type="train_stack",
            process_name="test_process",
            config={"a": 1},
            run_id="run_1",
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @patch("orchestration.handlers.train_stack.plot_history")
    @patch("orchestration.handlers.train_stack.save_json")
    @patch("orchestration.handlers.train_stack.train_stack")
    def test_run_train_stack_populates_best_metric_and_epoch(
        self,
        train_stack_mock: MagicMock,
        save_json_mock: MagicMock,
        plot_history_mock: MagicMock,
    ) -> None:
        train_stack_mock.return_value = {
            "mode": "mlp",
            "history": {"val": {"contest_metric": [0.11, 0.65, 0.5]}},
            "weights_path": str(self.run_dir / "weights" / "stack_mlp.pt"),
            "feature_spec": {"version": "1.0"},
        }

        weights, metrics, outputs = run_train_stack(self.spec, self.run_dir)

        self.assertIn("stack", weights)
        self.assertEqual(metrics["best_metric"], 0.65)
        self.assertEqual(metrics["best_epoch"], 2)
        self.assertEqual(outputs["mode"], "mlp")
        self.assertTrue(save_json_mock.called)
        self.assertTrue(plot_history_mock.called)

    @patch("orchestration.handlers.train_stack.plot_history")
    @patch("orchestration.handlers.train_stack.save_json")
    @patch("orchestration.handlers.train_stack.train_stack")
    def test_run_train_stack_ridge_uses_train_metrics(
        self,
        train_stack_mock: MagicMock,
        _save_json_mock: MagicMock,
        _plot_history_mock: MagicMock,
    ) -> None:
        train_stack_mock.return_value = {
            "mode": "ridge",
            "history": {"train": {"contest_metric": [0.41]}},
            "weights_path": str(self.run_dir / "weights" / "ridge.joblib"),
            "feature_spec": {},
        }

        _weights, metrics, _outputs = run_train_stack(self.spec, self.run_dir)
        self.assertEqual(metrics["best_metric"], 0.41)
        self.assertEqual(metrics["best_epoch"], 1)

    @patch("orchestration.handlers.train_stack.plot_history")
    @patch("orchestration.handlers.train_stack.save_json")
    @patch("orchestration.handlers.train_stack.train_stack")
    def test_run_train_stack_without_metric_history_returns_none_metric(
        self,
        train_stack_mock: MagicMock,
        save_json_mock: MagicMock,
        plot_history_mock: MagicMock,
    ) -> None:
        train_stack_mock.return_value = {
            "mode": "mlp",
            "history": {"val": {"contest_metric": []}},
            "weights_path": str(self.run_dir / "weights" / "stack_mlp.pt"),
            "feature_spec": {"version": "1.0"},
        }
        _weights, metrics, _outputs = run_train_stack(self.spec, self.run_dir)
        self.assertIn("best_metric", metrics)
        self.assertIsNone(metrics["best_metric"])
        self.assertNotIn("best_epoch", metrics)
        self.assertTrue(save_json_mock.called)
        self.assertTrue(plot_history_mock.called)

    @patch("orchestration.handlers.train_meta.train_meta_oof")
    def test_run_train_meta_maps_score_to_best_metric(
        self, train_meta_mock: MagicMock
    ) -> None:
        meta_spec = JobSpec(
            job_type="train_meta",
            process_name="meta",
            config={"variant": "lstm_ssm"},
            run_id="meta_1",
        )
        train_meta_mock.return_value = {
            "score": 0.1234,
            "weights_path": str(self.run_dir / "weights" / "meta_head_oof.pt"),
        }
        weights, metrics, outputs = run_train_meta(meta_spec, self.run_dir)
        self.assertIn("meta_head", weights)
        self.assertEqual(metrics["best_metric"], 0.1234)
        self.assertIn("score", outputs)

    @patch("orchestration.handlers.base_train.plot_history")
    @patch("orchestration.handlers.base_train.save_json")
    @patch("orchestration.handlers.base_train.run_training_job")
    def test_run_base_train_returns_expected_contract(
        self,
        training_job_mock: MagicMock,
        save_json_mock: MagicMock,
        plot_history_mock: MagicMock,
    ) -> None:
        fake_result = MagicMock()
        fake_result.best_weights_path = str(self.run_dir / "weights" / "best.pt")
        fake_result.best_metric = 0.987
        fake_result.best_epoch = 7
        fake_result.history = {"train": {"contest_metric": [0.987]}}
        training_job_mock.return_value = (fake_result, self.run_dir)

        base_spec = JobSpec(
            job_type="base_train",
            process_name="base",
            config={"model": {"name": "gru"}},
            run_id="base_1",
        )
        weights, metrics, outputs = run_base_train(base_spec, self.run_dir)
        self.assertEqual(weights["best"], fake_result.best_weights_path)
        self.assertEqual(metrics["best_metric"], fake_result.best_metric)
        self.assertEqual(outputs["best_epoch"], fake_result.best_epoch)
        self.assertTrue(save_json_mock.called)
        self.assertTrue(plot_history_mock.called)

    @patch("orchestration.handlers.build_oof.build_oof_dataset")
    def test_run_build_oof_passes_outputs_through(
        self, build_oof_mock: MagicMock
    ) -> None:
        build_oof_mock.return_value = {"merged": "x.parquet", "per_model": {"gru": "y.parquet"}}
        spec = JobSpec(
            job_type="build_oof",
            process_name="oof",
            config={"folds": 5},
            run_id="oof_1",
        )
        weights, metrics, outputs = run_build_oof(spec, self.run_dir)
        self.assertEqual(weights, {})
        self.assertEqual(metrics, {})
        self.assertEqual(outputs["merged"], "x.parquet")

    @patch("orchestration.handlers.optuna_tune.save_json")
    @patch("orchestration.handlers.optuna_tune.save_best_optuna_result")
    @patch("orchestration.handlers.optuna_tune.run_optuna_study")
    def test_run_optuna_tune_returns_best_metric_for_study_mode(
        self,
        run_study_mock: MagicMock,
        save_best_mock: MagicMock,
        save_json_mock: MagicMock,
    ) -> None:
        study = MagicMock()
        study.study_name = "st1"
        study.best_value = 0.81
        study.best_trial.number = 3
        run_study_mock.return_value = study
        save_best_mock.return_value = self.run_dir / "best_cfg.json"
        spec = JobSpec(
            job_type="optuna_tune",
            process_name="optuna",
            config={"model_name": "gru", "n_trials": 2},
            config_path=self.run_dir / "base.json",
            run_id="op_1",
        )
        (self.run_dir / "base.json").write_text("{}", encoding="utf-8")

        weights, metrics, outputs = run_optuna_tune(spec, self.run_dir)
        self.assertEqual(weights, {})
        self.assertEqual(metrics["best_metric"], 0.81)
        self.assertEqual(outputs["best_trial_number"], 3)
        self.assertTrue(save_json_mock.called)

    @patch("orchestration.handlers.optuna_tune.export_best_config_from_storage")
    def test_run_optuna_tune_export_only_validates_required_fields(
        self, export_mock: MagicMock
    ) -> None:
        export_mock.return_value = {"train_ready_config_path": "out.json"}
        spec = JobSpec(
            job_type="optuna_tune",
            process_name="optuna",
            config={
                "model_name": "gru",
                "export_only": True,
                "storage": "sqlite:///x.db",
                "study_name": "study_1",
            },
            config_path=self.run_dir / "base.json",
            run_id="op_2",
        )
        (self.run_dir / "base.json").write_text("{}", encoding="utf-8")
        weights, metrics, outputs = run_optuna_tune(spec, self.run_dir)
        self.assertEqual(weights, {})
        self.assertEqual(metrics, {})
        self.assertIn("train_ready_config_path", outputs)

    def test_run_optuna_tune_export_only_without_storage_raises(self) -> None:
        spec = JobSpec(
            job_type="optuna_tune",
            process_name="optuna",
            config={
                "model_name": "gru",
                "export_only": True,
                "study_name": "study_1",
            },
            config_path=self.run_dir / "base.json",
            run_id="op_3",
        )
        (self.run_dir / "base.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            run_optuna_tune(spec, self.run_dir)

    def test_run_optuna_tune_without_base_config_path_raises(self) -> None:
        spec = JobSpec(
            job_type="optuna_tune",
            process_name="optuna",
            config={"model_name": "gru", "n_trials": 1},
            config_path=None,
            run_id="op_4",
        )
        with self.assertRaises(ValueError):
            run_optuna_tune(spec, self.run_dir)


class TestJobsDispatcherAndManifest(unittest.TestCase):
    def test_ensure_run_dir_rejects_existing_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = root / "proc" / "run_1"
            (existing / "weights").mkdir(parents=True, exist_ok=True)
            with self.assertRaises(FileExistsError):
                _ensure_run_dir(root, "proc", "run_1", allow_overwrite=False)

    def test_ensure_run_dir_allows_existing_with_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = root / "proc" / "run_1"
            existing.mkdir(parents=True, exist_ok=True)
            run_dir = _ensure_run_dir(root, "proc", "run_1", allow_overwrite=True)
            self.assertEqual(run_dir, existing)
            self.assertTrue((run_dir / "weights").exists())
            self.assertTrue((run_dir / "plots").exists())
            self.assertTrue((run_dir / "config_snapshot").exists())

    def test_execute_job_builds_manifest_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec = build_job_spec(
                job_type="base_train",
                process_name="train_gru",
                config={"epochs": 2},
                run_id="manual_1",
                artifacts_root=td,
            )
            fake_outputs = (
                {"best": "weights/best.pt"},
                {"best_metric": 0.44, "best_epoch": 2},
                {"weights_path": "weights/best.pt"},
            )
            with patch("orchestration.jobs._RUNNERS", {"base_train": lambda _s, _r: fake_outputs}):
                result = execute_job(spec)

            manifest_path = Path(result.manifest_path)
            self.assertTrue(manifest_path.exists())
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["job_type"], "base_train")
            self.assertEqual(payload["run_id"], "manual_1")
            self.assertEqual(payload["metrics"]["best_metric"], 0.44)
            self.assertEqual(result.best_metric, 0.44)
            self.assertEqual(result.best_epoch, 2)

    def test_execute_job_raises_on_unsupported_job_type(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_spec = JobSpec(
                job_type="base_train",
                process_name="x",
                config={},
                run_id="z1",
                artifacts_root=td,
            )
            with patch("orchestration.jobs._RUNNERS", {}):
                with self.assertRaises(ValueError):
                    execute_job(bad_spec)

    def test_build_job_spec_preserves_optional_fields(self) -> None:
        spec = build_job_spec(
            job_type="build_oof",
            process_name="oof_build",
            config={"folds": 5},
            config_path="configs/build_oof.json",
            run_id="rid_1",
            artifacts_root="/tmp/artifacts",
            output_manifest="/tmp/custom_manifest.json",
            write_latest_link=True,
        )
        self.assertEqual(spec.job_type, "build_oof")
        self.assertEqual(spec.process_name, "oof_build")
        self.assertEqual(spec.config["folds"], 5)
        self.assertEqual(str(spec.output_manifest), "/tmp/custom_manifest.json")
        self.assertTrue(spec.write_latest_link)

    def test_execute_job_saves_inline_config_when_config_path_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec = build_job_spec(
                job_type="base_train",
                process_name="train_inline_cfg",
                config={"epochs": 1, "lr": 0.01},
                run_id="inline_1",
                artifacts_root=td,
            )
            fake_outputs = (
                {"best": "weights/best.pt"},
                {"best_metric": 0.2},
                {"weights_path": "weights/best.pt"},
            )
            with patch("orchestration.jobs._RUNNERS", {"base_train": lambda _s, _r: fake_outputs}):
                result = execute_job(spec)
            run_dir = Path(result.run_dir)
            config_path = run_dir / "config.json"
            self.assertTrue(config_path.exists())
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["epochs"], 1)

    @patch("orchestration.jobs.update_latest_link")
    def test_execute_job_updates_latest_link_when_enabled(
        self, update_latest_link_mock: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            spec = build_job_spec(
                job_type="base_train",
                process_name="train_gru",
                config={"epochs": 2},
                run_id="manual_2",
                artifacts_root=td,
                write_latest_link=True,
            )
            fake_outputs = (
                {"best": "weights/best.pt"},
                {"best_metric": 0.55},
                {"weights_path": "weights/best.pt"},
            )
            with patch("orchestration.jobs._RUNNERS", {"base_train": lambda _s, _r: fake_outputs}):
                _ = execute_job(spec)
            self.assertTrue(update_latest_link_mock.called)


if __name__ == "__main__":
    unittest.main()
