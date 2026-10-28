from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from med_ocr_research.evaluation.seed_metrics import aggregate_seed_metrics, read_seed_metrics, write_seed_summary
from med_ocr_research.evaluation.provenance import checkpoint_fingerprint
from med_ocr_research.experiments.migration import migrate_seed_runs
from med_ocr_research.experiments.seeds import (
    PROJECT_ROOT, Experiment, SeedSettings, ensure_new_run, load_settings,
    seeded_run_name, training_command, training_complete,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SeedWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.experiment = Experiment("e1.03", "cnn", "combined")
        self.settings = SeedSettings(root, (42, 123, 456), 42, (self.experiment,), root / "outputs/runs", root / "outputs/metrics")
        self.test_csv = root / "data/splits/scenarios/combined/test.csv"
        write_csv(self.test_csv, [{"sample_id": "a", "text_normalized": "abc", "split": "test"}])
        self.reference = {
            "epochs": "200", "batch_size": "16", "learning_rate": "0.0001", "weight_decay": "0.01",
            "num_workers": "2", "image_height": "64", "image_width": "512",
            "label_column": "text_normalized", "image_column": "image_path",
            "augmentation_enabled": "True", "augmentation_strength": "strong",
            "train_csv": str(self.test_csv.parent / "train.csv"),
            "val_csv": str(self.test_csv.parent / "val.csv"), "data_root": str(root / "data"),
        }
        for split in ("train", "val"):
            write_csv(self.test_csv.parent / f"{split}.csv", [{"sample_id": "x", "text_normalized": "abc", "split": split}])

    def record(self, seed: int, accuracy: float = 0.5, cer: float = 0.2, wer: float = 0.3) -> Path:
        experiment, settings = self.experiment, self.settings
        run_dir = settings.runs_dir / experiment.name(seed)
        write_csv(run_dir / "run_config.csv", [{**self.reference, "run_name": experiment.name(seed), "seed": str(seed)}])
        write_csv(run_dir / "run_summary.csv", [{"epochs_completed": "200", "run_name": experiment.name(seed)}])
        experiment.checkpoint(settings.runs_dir, seed).write_bytes(b"unchanged checkpoint")
        directory = settings.metrics_dir / experiment.evaluation_name(seed)
        directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_name": experiment.evaluation_name(seed), "source": str(experiment.checkpoint(settings.runs_dir, seed)),
            "test_csv": str(self.test_csv),
            "metrics": {"samples": 1, "exact_match_accuracy": accuracy, "cer": cer, "wer": wer},
        }
        (directory / "metrics.json").write_text(json.dumps(summary))
        write_csv(directory / "evaluated_predictions.csv", [{"sample_id": "a", "text_true": "abc", "text_pred": "abc"}])
        return directory

    def test_config_lists_existing_and_two_additional_seeds(self) -> None:
        settings = load_settings()
        self.assertEqual(settings.seeds, (42, 123, 456))
        self.assertEqual(settings.additional_seeds, (123, 456))
        self.assertEqual(len(settings.experiments) * len(settings.additional_seeds), 24)

    def test_names_reject_conflicting_seed_and_existing_artifacts(self) -> None:
        self.assertEqual(seeded_run_name("e1.03", 123), "e1.03_seed123")
        self.assertEqual(seeded_run_name("e1.03_seed123", 123), "e1.03_seed123")
        with self.assertRaises(ValueError):
            seeded_run_name("e1.03_seed42", 123)
        self.record(42)
        with self.assertRaises(FileExistsError):
            ensure_new_run(self.settings.runs_dir / "e1.03_seed42")

    def test_completed_runs_require_matching_parameters_and_weights(self) -> None:
        self.record(42)
        self.assertTrue(training_complete(self.settings, self.experiment, 42, self.reference))
        with self.assertRaisesRegex(ValueError, "batch_size"):
            training_complete(self.settings, self.experiment, 42, {**self.reference, "batch_size": "2"})
        self.experiment.checkpoint(self.settings.runs_dir, 42).unlink()
        self.assertFalse(training_complete(self.settings, self.experiment, 42, self.reference))

    def test_commands_preserve_transformer_reference_instead_of_defaults(self) -> None:
        reference = {**self.reference, "max_target_length": "64", "max_generation_length": "64",
                     "num_beams": "1", "gradient_accumulation_steps": "16", "warmup_ratio": "0.1",
                     "max_grad_norm": "1.0", "fp16": "False", "gradient_checkpointing": "False",
                     "freeze_encoder": "False", "checkpoint": "microsoft/trocr-small-handwritten",
                     "processor_checkpoint": "microsoft/trocr-small-handwritten", "pretrained": "True",
                     "charset": "abc ", "model_name": "parseq", "hub_repo": "baudm/parseq:fixed"}
        for model in ("cnn", "crnn", "trocr", "parseq"):
            with self.subTest(model=model):
                command = training_command(self.settings, Experiment("e1.03", model, "combined"), 123, reference, "python3")
                self.assertEqual(command[command.index("--seed") + 1], "123")
                self.assertEqual(command[command.index("--batch-size") + 1], "16")
                self.assertEqual(command[command.index("--epochs") + 1], "200")
                if model in {"trocr", "parseq"}:
                    self.assertIn("--no-fp16", command)
                    self.assertIn("--no-gradient-checkpointing", command)
                    self.assertEqual(command[command.index("--gradient-accumulation-steps") + 1], "16")
                if model == "trocr":
                    self.assertEqual(command[command.index("--checkpoint") + 1], reference["checkpoint"])
                if model == "parseq":
                    self.assertEqual(command[command.index("--extra-characters") + 1], "abc ")

    def test_median_is_computed_independently_for_each_metric(self) -> None:
        self.record(42, 0.1, 0.9, 0.2)
        self.record(123, 0.7, 0.1, 0.8)
        self.record(456, 0.9, 0.2, 0.6)
        summary = aggregate_seed_metrics(self.settings)
        median = summary["medians"][0]
        self.assertEqual(median["exact_match_accuracy_median"], 0.7)
        self.assertEqual(median["cer_median"], 0.2)
        self.assertEqual(median["wer_median"], 0.6)
        self.assertEqual(len(summary["per_seed"]), 3)
        destination = self.settings.metrics_dir / "seed_summary"
        write_seed_summary(summary, destination)
        self.assertEqual(len(list(destination.iterdir())), 3)

    def test_partial_results_do_not_produce_final_median(self) -> None:
        self.record(42)
        with self.assertRaisesRegex(FileNotFoundError, "seed123"):
            aggregate_seed_metrics(self.settings)
        self.assertFalse((self.settings.metrics_dir / "seed_summary").exists())

    def test_different_training_parameters_cannot_enter_final_median(self) -> None:
        for seed in self.settings.seeds:
            self.record(seed)
        config_path = self.settings.runs_dir / "e1.03_seed123/run_config.csv"
        with config_path.open() as stream:
            row = next(csv.DictReader(stream))
        row["batch_size"] = "2"
        write_csv(config_path, [row])
        with self.assertRaisesRegex(ValueError, "batch_size"):
            aggregate_seed_metrics(self.settings)

    def test_same_count_with_different_samples_or_labels_is_rejected(self) -> None:
        directory = self.record(42)
        for sample_id, text in [("other", "abc"), ("a", "different")]:
            with self.subTest(sample_id=sample_id, text=text):
                write_csv(directory / "evaluated_predictions.csv", [{"sample_id": sample_id, "text_true": text}])
                with self.assertRaisesRegex(ValueError, "teste completo"):
                    read_seed_metrics(self.settings, self.experiment, 42)

    def test_wrong_checkpoint_and_nonfinite_metrics_are_rejected(self) -> None:
        directory = self.record(42)
        path = directory / "metrics.json"
        summary = json.loads(path.read_text())
        summary["source"] = str(self.experiment.checkpoint(self.settings.runs_dir, 123))
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            read_seed_metrics(self.settings, self.experiment, 42)
        self.record(42, cer=float("nan"))
        with self.assertRaisesRegex(ValueError, "nao finitas"):
            read_seed_metrics(self.settings, self.experiment, 42)

    def test_current_protocol_rejects_stale_evaluations_even_at_same_checkpoint_path(self) -> None:
        provenance = {"protocol_version": "equivalent-v2", "protocol_fingerprint": "current-splits-and-protocol"}
        self.reference.update(provenance)
        self.settings = replace(self.settings, protocol={"protocol_version": "equivalent-v2"})
        directory = self.record(42)
        path = directory / "metrics.json"
        summary = json.loads(path.read_text())
        summary.update(provenance, training_seed=42,
                       checkpoint_fingerprint=checkpoint_fingerprint(self.experiment.checkpoint(self.settings.runs_dir, 42)))
        with patch("med_ocr_research.evaluation.seed_metrics.baseline_config", return_value=self.reference):
            path.write_text(json.dumps(summary))
            self.assertEqual(read_seed_metrics(self.settings, self.experiment, 42)["seed"], 42)
            for field, value in (("protocol_version", None), ("protocol_fingerprint", "old-protocol"),
                                 ("training_seed", 123), ("checkpoint_fingerprint", None)):
                with self.subTest(field=field):
                    path.write_text(json.dumps({**summary, field: value}))
                    with self.assertRaises(ValueError):
                        read_seed_metrics(self.settings, self.experiment, 42)
            path.write_text(json.dumps(summary))
            self.experiment.checkpoint(self.settings.runs_dir, 42).write_bytes(b"new training weights at same path")
            with self.assertRaisesRegex(ValueError, "Pesos do checkpoint"):
                read_seed_metrics(self.settings, self.experiment, 42)

    def legacy_run(self) -> tuple[Path, Path]:
        run = self.settings.runs_dir / "e1.03"
        write_csv(run / "run_config.csv", [{"seed": "42", "run_name": "e1.03", "output_dir": str(run)}])
        (run / "e1cnn_best_checkpoint.pt").write_bytes(b"weights untouched")
        metrics = self.settings.metrics_dir / "e1.03cnn_evaluation"
        metrics.mkdir(parents=True)
        (metrics / "metrics.json").write_text(json.dumps({"run_name": metrics.name, "source": str(run / "e1cnn_best_checkpoint.pt"), "metrics": {"cer": 0.42}}))
        return run, metrics

    def test_migration_preserves_weights_metrics_and_is_idempotent(self) -> None:
        run, metrics = self.legacy_run()
        checkpoint = run / "e1cnn_best_checkpoint.pt"
        inode = checkpoint.stat().st_ino
        self.assertEqual(len(migrate_seed_runs(self.settings, dry_run=True)), 2)
        self.assertTrue(run.exists())
        self.assertEqual(len(migrate_seed_runs(self.settings)), 2)
        new_checkpoint = self.experiment.checkpoint(self.settings.runs_dir, 42)
        self.assertEqual(new_checkpoint.stat().st_ino, inode)
        self.assertEqual(new_checkpoint.read_bytes(), b"weights untouched")
        summary = json.loads((self.settings.metrics_dir / "e1.03_seed42_evaluation/metrics.json").read_text())
        self.assertEqual(summary["run_name"], "e1.03_seed42_evaluation")
        self.assertEqual(summary["source"], str(new_checkpoint))
        self.assertEqual(summary["metrics"]["cer"], 0.42)
        self.assertEqual(migrate_seed_runs(self.settings), [])

    def test_migration_detects_collision_before_changing_source(self) -> None:
        run, _ = self.legacy_run()
        (self.settings.metrics_dir / "e1.03_seed42_evaluation").mkdir()
        with self.assertRaises(FileExistsError):
            migrate_seed_runs(self.settings)
        self.assertTrue(run.exists())
        self.assertFalse((self.settings.runs_dir / "e1.03_seed42").exists())

    def test_migration_rolls_back_paths_and_metadata_if_a_move_fails(self) -> None:
        run, metrics = self.legacy_run()
        original = (run / "run_config.csv").read_bytes()
        rename = Path.rename

        def fail_second_move(path, destination):
            if path == metrics:
                raise OSError("simulated failure")
            return rename(path, destination)

        with patch.object(Path, "rename", fail_second_move), self.assertRaises(OSError):
            migrate_seed_runs(self.settings)
        self.assertTrue(run.exists())
        self.assertTrue(metrics.exists())
        self.assertEqual((run / "run_config.csv").read_bytes(), original)
        self.assertFalse((self.settings.runs_dir / "e1.03_seed42").exists())

    def runner(self):
        spec = importlib.util.spec_from_file_location("seed_runner", PROJECT_ROOT / "scripts/train_all_seeds.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_dry_run_does_not_start_training_or_write_artifacts(self) -> None:
        self.record(42)
        runner = self.runner()
        before = sorted(str(path) for path in self.settings.root.rglob("*"))
        with patch.object(runner, "load_settings", return_value=self.settings), patch.object(sys, "argv", ["runner", "--dry-run"]), patch.object(runner.subprocess, "run") as run, patch("builtins.print"):
            runner.main()
        run.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.settings.root.rglob("*")))

    def test_training_failure_stops_before_evaluation_or_aggregation(self) -> None:
        self.record(42)
        runner = self.runner()
        with patch.object(runner, "load_settings", return_value=self.settings), patch.object(sys, "argv", ["runner"]), patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "train")) as run, patch("builtins.print"):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main()
        self.assertEqual(run.call_count, 1)
        self.assertIn("train_e1cnn.py", run.call_args.args[0][1])
        self.assertFalse((self.settings.metrics_dir / "seed_summary").exists())

    def test_stale_metrics_schedule_reevaluation_without_retraining_completed_models(self) -> None:
        for seed in self.settings.seeds:
            self.record(seed)
        runner = self.runner()

        def validate_metrics(settings, experiment, seed):
            if seed == 123:
                raise ValueError("checkpoint fingerprint changed")
            return read_seed_metrics(settings, experiment, seed)

        with patch.object(runner, "load_settings", return_value=self.settings), patch.object(sys, "argv", ["runner"]), \
             patch.object(runner, "read_seed_metrics", side_effect=validate_metrics), \
             patch.object(runner.subprocess, "run") as run, patch("builtins.print"):
            runner.main()
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("evaluate_model.py", command[1])
        self.assertEqual(command[command.index("--run-name") + 1], "e1.03_seed123_evaluation")

    def test_runner_finishes_new_seeds_and_writes_medians_without_retraining_baseline(self) -> None:
        self.record(42, accuracy=0.1)
        runner = self.runner()

        def finish_job(command, **kwargs):
            if "--seed" in command:
                seed = int(command[command.index("--seed") + 1])
                self.assertNotEqual(seed, 42)
                self.record(seed, accuracy={123: 0.9, 456: 0.7}[seed])

        with patch.object(runner, "load_settings", return_value=self.settings), patch.object(sys, "argv", ["runner"]), patch.object(runner.subprocess, "run", side_effect=finish_job) as run, patch("builtins.print"):
            runner.main()
        self.assertEqual(run.call_count, 4)
        result = json.loads((self.settings.metrics_dir / "seed_summary/seed_medians.json").read_text())
        self.assertEqual(result["training_seeds"], [42, 123, 456])
        self.assertEqual(result["medians"][0]["exact_match_accuracy_median"], 0.7)


if __name__ == "__main__":
    unittest.main()
