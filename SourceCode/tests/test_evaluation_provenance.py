from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from med_ocr_research.evaluation.evaluate_model import build_summary, enrich_predictions
from med_ocr_research.evaluation.provenance import checkpoint_fingerprint, training_provenance


def write_run_config(directory: Path) -> None:
    with (directory / "run_config.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "protocol_version", "protocol_fingerprint"])
        writer.writeheader()
        writer.writerow({"seed": 123, "protocol_version": "equivalent-v2", "protocol_fingerprint": "protocol-content"})


def test_file_fingerprint_detects_retraining_at_same_path(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    source.write_bytes(b"old weights")
    before = checkpoint_fingerprint(source)
    assert before == hashlib.sha256(b"old weights").hexdigest()
    source.write_bytes(b"new weights")
    assert checkpoint_fingerprint(source) != before


def test_bundle_fingerprint_is_deterministic_and_ignores_nonweights(tmp_path: Path) -> None:
    original = tmp_path / "original"
    moved = tmp_path / "moved"
    original.mkdir()
    moved.mkdir()
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name in names:
        (original / name).write_bytes(name.encode())
    for name in reversed(names):
        (moved / name).write_bytes(name.encode())
    expected = checkpoint_fingerprint(original)
    assert checkpoint_fingerprint(moved) == expected
    for name in ["config.json", "tokenizer.json", "tokenizer.pt", "optimizer.pt", "scheduler.pt"]:
        (moved / name).write_bytes(b"unrelated")
    assert checkpoint_fingerprint(moved) == expected
    (moved / names[0]).write_bytes(b"changed weight shard")
    assert checkpoint_fingerprint(moved) != expected


@pytest.mark.parametrize("weight_name", ["model.pt", "model.safetensors", "pytorch_model.bin"])
def test_training_provenance_reads_adjacent_bundle_configuration(tmp_path: Path, weight_name: str) -> None:
    bundle = tmp_path / "best"
    bundle.mkdir()
    (bundle / weight_name).write_bytes(b"weights")
    write_run_config(tmp_path)

    provenance = training_provenance(bundle)

    assert provenance == {
        "training_seed": 123,
        "protocol_version": "equivalent-v2",
        "protocol_fingerprint": "protocol-content",
        "checkpoint_fingerprint": checkpoint_fingerprint(bundle),
    }


def test_provenance_supports_cnn_checkpoint_and_standalone_inputs(tmp_path: Path) -> None:
    source = tmp_path / "best.pt"
    source.write_bytes(b"weights")
    assert training_provenance(source) == {}
    assert training_provenance(None) == {}
    write_run_config(tmp_path)
    assert training_provenance(source)["training_seed"] == 123
    assert training_provenance(source)["checkpoint_fingerprint"] == hashlib.sha256(b"weights").hexdigest()


def test_missing_weight_bundle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="pesos"):
        checkpoint_fingerprint(tmp_path)


@pytest.mark.parametrize("from_checkpoint", [False, True])
def test_shared_summary_records_provenance_only_for_trained_checkpoint(tmp_path: Path, from_checkpoint: bool) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    write_run_config(tmp_path)
    args = SimpleNamespace(
        predictions_csv=None if from_checkpoint else tmp_path / "predictions.csv",
        checkpoint=checkpoint if from_checkpoint else None,
        test_csv=tmp_path / "test.csv",
    )
    predictions = enrich_predictions(pd.DataFrame([{
        "sample_id": "1", "dataset": "fixture", "text_true": "abc", "text_pred": "ab",
    }]))

    summary = build_summary(args, "example", predictions)

    assert summary["metrics"]["samples"] == 1
    assert ("protocol_fingerprint" in summary) is from_checkpoint
    assert ("checkpoint_fingerprint" in summary) is from_checkpoint
    if from_checkpoint:
        assert summary["training_seed"] == 123
