from __future__ import annotations

import http.client
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import yaml

from med_ocr_research.experiments.monitor import MonitorServer, access_token, build_snapshot, log_tail
from med_ocr_research.experiments import monitor
from med_ocr_research.experiments.progress import PipelineProgress, pipeline_status, process_identity, runtime_directory
from med_ocr_research.experiments.seeds import DEFAULT_CONFIG, PROJECT_ROOT, Experiment, SeedSettings, baseline_config


class TrainingMonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = runtime_directory(self.root)
        self.settings = SeedSettings(self.root, (42, 123, 456), 42,
                                     (Experiment("e1.03", "cnn", "combined"),),
                                     self.root / "outputs/runs", self.root / "outputs/metrics")

    def test_pipeline_records_progress_and_prevents_concurrent_execution(self):
        with PipelineProgress(self.directory, self.settings.seeds) as tracker:
            tracker.update(status="running", phase="training", current_run="e1.03_seed123", total_steps=4)
            with self.assertRaisesRegex(RuntimeError, "ativo"):
                with PipelineProgress(self.directory, self.settings.seeds):
                    pass
            self.assertEqual(pipeline_status(self.directory)["current_run"], "e1.03_seed123")
        self.assertEqual(pipeline_status(self.directory)["status"], "completed")
        with PipelineProgress(self.directory, self.settings.seeds):
            pass

    def test_failure_keeps_current_run_and_return_code(self):
        with self.assertRaises(subprocess.CalledProcessError):
            with PipelineProgress(self.directory, self.settings.seeds) as tracker:
                tracker.update(phase="evaluation", current_run="e1.03_seed123")
                raise subprocess.CalledProcessError(7, "evaluate")
        state = pipeline_status(self.directory)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["exit_code"], 7)
        self.assertEqual(state["current_run"], "e1.03_seed123")

    def test_stale_process_is_reported_as_interrupted(self):
        with PipelineProgress(self.directory, self.settings.seeds):
            pass
        path = self.directory / "pipeline.json"
        state = json.loads(path.read_text())
        state.update(status="running", process={**process_identity(os.getpid()), "start_ticks": "different"})
        path.write_text(json.dumps(state))
        self.assertEqual(pipeline_status(self.directory)["status"], "interrupted")

    def test_sigterm_records_stopped_state_in_real_subprocess(self):
        code = (
            "import sys,time\nfrom pathlib import Path\n"
            "from med_ocr_research.experiments.progress import PipelineProgress\n"
            "with PipelineProgress(Path(sys.argv[1]), (42,123,456)) as tracker:\n"
            " tracker.update(status='running', phase='training', current_run='synthetic_test')\n"
            " print('ready',flush=True)\n time.sleep(30)\n"
        )
        process = subprocess.Popen([sys.executable, "-c", code, str(self.directory)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            process.terminate()
            process.communicate(timeout=5)
            self.assertEqual(pipeline_status(self.directory)["status"], "stopped")
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_token_is_persistent_and_private(self):
        token = access_token(self.directory)
        self.assertEqual(token, access_token(self.directory))
        self.assertGreaterEqual(len(token), 32)
        self.assertEqual((self.directory / "access_token").stat().st_mode & 0o777, 0o600)

    def test_snapshot_tolerates_files_being_written_and_bounds_log(self):
        run_dir = self.settings.runs_dir / "e1.03_seed123"
        run_dir.mkdir(parents=True)
        (run_dir / "run_config.csv").write_text("epochs\n200\n")
        (run_dir / "training_history.csv").write_text("epoch,val_loss,val_accuracy\n1,0.4,0.8\n2,")
        metric_dir = self.settings.metrics_dir / "e1.03_seed123_evaluation"
        metric_dir.mkdir(parents=True)
        (metric_dir / "metrics.json").write_text('{"metrics":')
        self.directory.mkdir(parents=True)
        (self.directory / "training.log").write_text("x" * 100000 + "\nlatest line\n")
        snapshot = build_snapshot(self.settings)
        self.assertEqual(len(snapshot["runs"]), 3)
        self.assertEqual(snapshot["evaluated"], 0)
        self.assertEqual(snapshot["runs"][1]["status"], "partial")
        self.assertLess(len(snapshot["log"]), 49000)
        self.assertIn("latest line", log_tail(self.directory / "training.log"))

    def test_http_authentication_readonly_routes_and_live_status(self):
        server = MonitorServer(("127.0.0.1", 0), self.settings, "test-token")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        try:
            for route, code, headers in [("/", 200, {}), ("/api/status", 401, {}),
                                          ("/api/status", 401, {"Authorization": "Bearer wrong"}),
                                          ("/../../configs/default.yaml", 404, {"Authorization": "Bearer test-token"})]:
                connection.request("GET", route, headers=headers)
                response = connection.getresponse()
                self.assertEqual(response.status, code)
                response.read()
            connection.request("GET", "/api/status", headers={"Authorization": "Bearer test-token"})
            response = connection.getresponse()
            snapshot = json.loads(response.read())
            self.assertEqual(snapshot["pipeline"]["status"], "idle")
            self.assertEqual(snapshot["seeds"], [42, 123, 456])
            connection.request("POST", "/api/start", body=b"")
            response = connection.getresponse()
            self.assertEqual(response.status, 501)
            response.read()
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_systemd_units_are_separate_and_training_does_not_autorestart(self):
        spec = importlib.util.spec_from_file_location("training_service", PROJECT_ROOT / "scripts/training_service.py")
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)
        command = ["/path with space/.venv/bin/python", "-u", "/path with space/train.py"]
        text = service.service_text(command, self.root, self.directory / "training.log", monitor=False)
        self.assertIn('ExecStart=:"/path with space/.venv/bin/python"', text)
        self.assertIn("KillMode=control-group", text)
        self.assertIn("Restart=no", text)
        self.assertIn(f"WorkingDirectory={self.root}\n", text)
        monitor = service.service_text(command, self.root, self.directory / "monitor.log", monitor=True)
        self.assertIn("Restart=on-failure", monitor)
        self.assertNotEqual(*service.unit_names(self.root))


class EquivalentProtocolMonitorTests(unittest.TestCase):
    def setUp(self):
        TrainingMonitorTests.setUp(self)
        protocol = yaml.safe_load(DEFAULT_CONFIG.read_text())["training"]
        self.settings = replace(self.settings, protocol=protocol)
        for split in ("train", "val", "test"):
            self.write_csv(self.root / "data/splits/scenarios/combined" / f"{split}.csv", [{
                "sample_id": f"{split}_sample", "split": split,
                "image_path": f"{split}.png", "text_normalized": "example",
            }])
        self.experiment = self.settings.experiments[0]
        reference = baseline_config(self.settings, self.experiment)
        per_seed = []
        for seed in self.settings.seeds:
            run = self.settings.runs_dir / self.experiment.name(seed)
            self.write_csv(run / "run_config.csv", [{**reference, "run_name": run.name, "seed": seed}])
            self.write_csv(run / "run_summary.csv", [{"epochs_completed": 200}])
            checkpoint = self.experiment.checkpoint(self.settings.runs_dir, seed)
            checkpoint.write_bytes(b"weights")
            metrics = {"samples": 1, "exact_match_accuracy": 0.5, "cer": 0.2, "wer": 0.4}
            provenance = {"protocol_version": reference["protocol_version"],
                          "protocol_fingerprint": reference["protocol_fingerprint"],
                          "checkpoint_fingerprint": f"weights-{seed}"}
            evaluation = {"run_name": self.experiment.evaluation_name(seed), "training_seed": seed,
                          "source": str(checkpoint), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                          "metrics": metrics, **provenance}
            metric_path = self.metrics_path(seed)
            metric_path.parent.mkdir(parents=True)
            metric_path.write_text(json.dumps(evaluation))
            per_seed.append({"base_run_name": self.experiment.run_name, "seed": seed, **metrics, **provenance})
        self.final_path = self.settings.metrics_dir / "seed_summary/seed_medians.json"
        self.final_path.parent.mkdir(parents=True)
        self.final_path.write_text(json.dumps({
            "protocol_version": protocol["protocol_version"], "training_protocol": protocol,
            "training_seeds": list(self.settings.seeds), "per_seed": per_seed,
            "medians": [{"base_run_name": self.experiment.run_name, "cer_median": 0.2}],
        }))

    @staticmethod
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def metrics_path(self, seed):
        return self.settings.metrics_dir / self.experiment.evaluation_name(seed) / "metrics.json"

    def test_current_results_and_medians_are_visible(self):
        snapshot = build_snapshot(self.settings)
        self.assertEqual(snapshot["trained"], 3)
        self.assertEqual(snapshot["evaluated"], 3)
        self.assertEqual(len(snapshot["medians"]), 1)

    def test_reference_cache_reuses_validation_and_invalidates_on_split_or_config_change(self):
        with patch.object(monitor, "baseline_config", wraps=baseline_config) as validate:
            first = build_snapshot(self.settings)
            second = build_snapshot(self.settings)
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(first["evaluated"], second["evaluated"])
            path = self.root / "data/splits/scenarios/combined/val.csv"
            stat = path.stat()
            # Preserve both size and mtime: ctime still reveals this rewrite.
            path.write_text(path.read_text().replace("example", "changed"))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            stale = build_snapshot(self.settings)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(stale["evaluated"], 0)
            changed = replace(self.settings, protocol={**self.settings.protocol, "max_epochs": 201})
            build_snapshot(changed)
            self.assertEqual(validate.call_count, 3)
            build_snapshot(changed)
            self.assertEqual(validate.call_count, 3)

    def test_stale_individual_protocol_metrics_hide_final_medians(self):
        path = self.metrics_path(42)
        metadata = json.loads(path.read_text())
        metadata["protocol_fingerprint"] = "old-fingerprint"
        path.write_text(json.dumps(metadata))
        snapshot = build_snapshot(self.settings)
        self.assertEqual(snapshot["trained"], 3)
        self.assertEqual(snapshot["evaluated"], 2)
        self.assertEqual(snapshot["runs"][0]["status"], "trained")
        self.assertIsNone(snapshot["runs"][0]["metrics"]["cer"])
        self.assertEqual(snapshot["medians"], [])

    def test_same_protocol_version_with_new_configuration_hides_old_results(self):
        settings = replace(self.settings, protocol={**self.settings.protocol, "max_epochs": 201})
        snapshot = build_snapshot(settings)
        self.assertEqual(snapshot["trained"], 0)
        self.assertEqual(snapshot["evaluated"], 0)
        self.assertEqual(snapshot["medians"], [])

    def test_changed_split_hides_results_without_a_version_bump(self):
        path = self.root / "data/splits/scenarios/combined/val.csv"
        path.write_text(path.read_text().replace("example", "different"))
        snapshot = build_snapshot(self.settings)
        self.assertEqual(snapshot["trained"], 0)
        self.assertEqual(snapshot["evaluated"], 0)
        self.assertEqual(snapshot["medians"], [])

    def test_same_version_medians_require_current_per_seed_weights(self):
        final = json.loads(self.final_path.read_text())
        final["per_seed"][0]["checkpoint_fingerprint"] = "old-weights"
        self.final_path.write_text(json.dumps(final))
        snapshot = build_snapshot(self.settings)
        self.assertEqual(snapshot["evaluated"], 3)
        self.assertEqual(snapshot["medians"], [])

    def test_checkpoint_replaced_after_evaluation_is_not_shown_as_evaluated(self):
        checkpoint = self.experiment.checkpoint(self.settings.runs_dir, 42)
        checkpoint.write_bytes(b"new weights")
        metadata = json.loads(self.metrics_path(42).read_text())
        future = datetime.fromisoformat(metadata["created_at_utc"]).timestamp() + 1
        os.utime(checkpoint, (future, future))
        snapshot = build_snapshot(self.settings)
        self.assertEqual(snapshot["evaluated"], 2)
        self.assertEqual(snapshot["medians"], [])


if __name__ == "__main__":
    unittest.main()
