from __future__ import annotations

import csv
import fcntl
import importlib.util
import json
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from med_ocr_research.experiments.archive import archive_incompatible_runs
from med_ocr_research.experiments.progress import runtime_directory
from med_ocr_research.experiments.seeds import (
    DEFAULT_CONFIG,
    DEFAULT_MATRIX,
    PROJECT_ROOT,
    SeedSettings,
    baseline_config,
    evaluation_command,
    load_settings,
    protocol_config,
    training_command,
    training_complete,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def equivalent_settings(tmp_path: Path) -> SeedSettings:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(DEFAULT_CONFIG.read_text())
    for dataset in ("prescriptionbd", "rxhandbd", "combined"):
        for split in ("train", "val", "test"):
            write_csv(tmp_path / "data/splits/scenarios" / dataset / f"{split}.csv", [
                {"sample_id": f"{dataset}_{split}_{index}", "dataset": dataset, "split": split,
                 "image_path": f"images/{dataset}_{split}_{index}.png",
                 "text_normalized": "known" if split == "train" else "unseen"}
                for index in range(2)
            ])
    return load_settings(config_path, DEFAULT_MATRIX, root=tmp_path)


def argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_central_protocol_applies_common_budget_and_scratch_initialization(equivalent_settings: SeedSettings) -> None:
    settings = equivalent_settings
    assert settings.seeds == (42, 123, 456)
    assert len(settings.experiments) == 12
    assert settings.protocol["pretrained_transformers"] is False
    expected = {
        "--epochs": "200", "--batch-size": "2", "--gradient-accumulation-steps": "8",
        "--learning-rate": "0.0001", "--weight-decay": "0.01", "--warmup-ratio": "0.1",
        "--max-grad-norm": "1.0", "--image-height": "64", "--image-width": "512",
        "--augmentation-strength": "strong", "--protocol-version": "equivalent-v2",
    }
    for experiment in settings.experiments:
        reference = baseline_config(settings, experiment)
        # No seed42 run directory or historical CSV is necessary to plan a run.
        assert not settings.runs_dir.exists()
        assert reference["effective_batch_size"] == "16"
        for seed in settings.seeds:
            command = training_command(settings, experiment, seed, reference, "training-python")
            for flag, value in expected.items():
                assert argument(command, flag) == value
            assert argument(command, "--seed") == str(seed)
            assert argument(command, "--run-name") == experiment.name(seed)
            assert argument(command, "--protocol-fingerprint") == reference["protocol_fingerprint"]
            assert "--fp16" in command
            assert "--pretrained" not in command
            if experiment.model in {"trocr", "parseq"}:
                assert "--no-pretrained" in command
                assert "--gradient-checkpointing" in command
                assert "--freeze-encoder" not in command
                assert argument(command, "--num-beams") == "1"


def test_runner_plans_all_36_retrainings_including_seed42(equivalent_settings: SeedSettings, capsys: pytest.CaptureFixture) -> None:
    spec = importlib.util.spec_from_file_location("equivalent_seed_runner", PROJECT_ROOT / "scripts/train_all_seeds.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.execute(SimpleNamespace(dry_run=True, python="training-python"), equivalent_settings)
    output = capsys.readouterr().out
    commands = [shlex.split(line) for line in output.splitlines() if line.startswith("training-python ")]
    training_commands = [command for command in commands if "--seed" in command]
    assert len(commands) == 72
    assert len(training_commands) == 36
    assert sum(argument(command, "--seed") == "42" for command in training_commands) == 12
    assert not equivalent_settings.runs_dir.exists()
    assert not equivalent_settings.metrics_dir.exists()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_fingerprint_tracks_every_split_and_complete_validation(equivalent_settings: SeedSettings, split: str) -> None:
    settings = equivalent_settings
    experiment = settings.experiments[0]
    before = protocol_config(settings, experiment)
    assert before["train_samples"] == "2"
    assert before["val_samples"] == "2"  # Both labels are absent from the training vocabulary.
    path = settings.root / "data/splits/scenarios" / experiment.dataset / f"{split}.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["text_normalized"] = "changed"
    write_csv(path, rows)
    after = protocol_config(settings, experiment)
    assert after["protocol_fingerprint"] != before["protocol_fingerprint"]
    assert after["val_samples"] == "2"


def test_fingerprint_changes_with_budget_or_pretraining(equivalent_settings: SeedSettings) -> None:
    settings = equivalent_settings
    experiment = settings.experiments[0]
    before = protocol_config(settings, experiment)["protocol_fingerprint"]
    for change in ({"max_epochs": 201}, {"pretrained_transformers": True}):
        changed = replace(settings, protocol={**settings.protocol, **change})
        assert protocol_config(changed, experiment)["protocol_fingerprint"] != before


@pytest.mark.parametrize("problem", ["duplicate_id", "overlapping_id", "empty_label", "empty_image"])
def test_protocol_rejects_invalid_or_overlapping_splits(equivalent_settings: SeedSettings, problem: str) -> None:
    settings = equivalent_settings
    experiment = settings.experiments[0]
    folder = settings.root / "data/splits/scenarios" / experiment.dataset
    path = folder / "val.csv"
    rows = list(csv.DictReader(path.open()))
    if problem == "duplicate_id":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif problem == "overlapping_id":
        train_rows = list(csv.DictReader((folder / "train.csv").open()))
        rows[0]["sample_id"] = train_rows[0]["sample_id"]
    elif problem == "empty_label":
        rows[0]["text_normalized"] = ""
    else:
        rows[0]["image_path"] = ""
    write_csv(path, rows)
    with pytest.raises(ValueError):
        protocol_config(settings, experiment)


def test_shared_evaluation_uses_same_complete_test_split(equivalent_settings: SeedSettings) -> None:
    for experiment in equivalent_settings.experiments:
        reference = baseline_config(equivalent_settings, experiment)
        command = evaluation_command(equivalent_settings, experiment, 42, reference, "python")
        assert "--limit" not in command
        assert argument(command, "--test-csv") == str(equivalent_settings.root / "data/splits/scenarios" / experiment.dataset / "test.csv")
        if experiment.model in {"trocr", "parseq"}:
            assert argument(command, "--num-beams") == "1"


def create_run(settings: SeedSettings, seed: int, *, compatible: bool) -> Path:
    experiment = settings.experiments[0]
    run = settings.runs_dir / experiment.name(seed)
    config = baseline_config(settings, experiment) if compatible else {"epochs": "200"}
    write_csv(run / "run_config.csv", [{**config, "seed": str(seed), "run_name": experiment.name(seed)}])
    write_csv(run / "run_summary.csv", [{"epochs_completed": "200"}])
    experiment.checkpoint(settings.runs_dir, seed).write_bytes(b"weights retained byte for byte")
    metrics = settings.metrics_dir / experiment.evaluation_name(seed)
    metrics.mkdir(parents=True)
    (metrics / "metrics.json").write_text('{"historical": true}')
    return run


def test_archive_preserves_legacy_artifacts_and_compatible_runs(equivalent_settings: SeedSettings) -> None:
    settings = replace(equivalent_settings, experiments=equivalent_settings.experiments[:1])
    old_run = create_run(settings, 42, compatible=False)
    current_run = create_run(settings, 123, compatible=True)
    runtime = runtime_directory(settings.root)
    runtime.mkdir(parents=True)
    for name in ("pipeline.json", "training.log", "access_token", "monitor.log"):
        (runtime / name).write_text(name)
    summary = settings.metrics_dir / "seed_summary"
    summary.mkdir()
    (summary / "seed_medians.json").write_text("historical medians")
    before = {str(path.relative_to(settings.root)): path.read_bytes() for path in settings.root.rglob("*") if path.is_file()}

    manifest = archive_incompatible_runs(settings)

    assert len(manifest["entries"]) == 5
    assert not old_run.exists()
    assert current_run.exists()
    assert (runtime / "access_token").read_text() == "access_token"
    assert (runtime / "monitor.log").read_text() == "monitor.log"
    for entry in manifest["entries"]:
        source = Path(entry["source"])
        archived = settings.root / entry["archived"]
        for original, data in before.items():
            original_path = Path(original)
            if original_path == source:
                assert archived.read_bytes() == data
            elif source in original_path.parents:
                assert (archived / original_path.relative_to(source)).read_bytes() == data
    persisted = json.loads(Path(manifest["manifest_path"]).read_text())
    assert persisted["new_protocol"] == settings.protocol["protocol_version"]
    assert archive_incompatible_runs(settings)["entries"] == []


def test_compatible_completed_run_is_preserved_without_archive(equivalent_settings: SeedSettings) -> None:
    settings = replace(equivalent_settings, experiments=equivalent_settings.experiments[:1])
    run = create_run(settings, 42, compatible=True)
    experiment = settings.experiments[0]
    assert training_complete(settings, experiment, 42, baseline_config(settings, experiment))
    manifest = archive_incompatible_runs(settings)
    assert manifest["entries"] == []
    assert run.exists()
    assert not (settings.root / "outputs/archive").exists()


def test_archive_dry_run_does_not_move_weights(equivalent_settings: SeedSettings) -> None:
    settings = replace(equivalent_settings, experiments=equivalent_settings.experiments[:1])
    run = create_run(settings, 42, compatible=False)
    manifest = archive_incompatible_runs(settings, dry_run=True)
    assert len(manifest["entries"]) == 2
    assert run.exists()
    assert not (settings.root / "outputs/archive").exists()


def test_archive_refuses_to_move_artifacts_during_training(equivalent_settings: SeedSettings) -> None:
    settings = replace(equivalent_settings, experiments=equivalent_settings.experiments[:1])
    run = create_run(settings, 42, compatible=False)
    runtime = runtime_directory(settings.root)
    runtime.mkdir(parents=True)
    with (runtime / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="treinamento em execucao"):
            archive_incompatible_runs(settings)
    assert run.exists()


def test_archive_rolls_back_already_moved_artifacts(equivalent_settings: SeedSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = replace(equivalent_settings, experiments=equivalent_settings.experiments[:1])
    run = create_run(settings, 42, compatible=False)
    original_rename = Path.rename

    def fail_second_move(source, destination):
        if source.name.endswith("_evaluation") and "archive" in Path(destination).parts:
            raise OSError("simulated interrupted archive")
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", fail_second_move)
    with pytest.raises(OSError, match="simulated"):
        archive_incompatible_runs(settings)
    assert run.exists()
    assert settings.experiments[0].checkpoint(settings.runs_dir, 42).read_bytes() == b"weights retained byte for byte"
