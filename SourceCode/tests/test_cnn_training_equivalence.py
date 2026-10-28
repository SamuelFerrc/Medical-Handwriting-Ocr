from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from med_ocr_research.training import train_e1cnn as training


def make_split(tmp_path: Path, labels: list[str]) -> Path:
    image = tmp_path / "sample.png"
    Image.new("L", (4, 4), "white").save(image)
    csv_path = tmp_path / "split.csv"
    pd.DataFrame({"image_path": [str(image)] * len(labels), "text_normalized": labels}).to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.parametrize("target_kind", [training.CLASSIFICATION_TARGET, training.CTC_TARGET])
def test_validation_keeps_unknown_targets(tmp_path: Path, target_kind: str) -> None:
    dataset = training.TrainingSplitDataset(
        make_split(tmp_path, ["a", "z"]), tmp_path, "text_normalized", "image_path", (4, 4),
        class_to_idx={"a": 0}, char_to_idx={"<blank>": 0, "a": 1},
        target_kind=target_kind, return_text=True,
    )

    assert len(dataset) == 2
    assert dataset.representable_samples == 1
    assert dataset[1][1].item() == -1
    assert dataset[1][2] == "z"


@pytest.mark.parametrize("labels, expected_samples", [(["a", "z"], 1), (["z", "y"], 0)])
def test_classification_metrics_include_unknown_words(tmp_path: Path, labels: list[str], expected_samples: int) -> None:
    dataset = training.TrainingSplitDataset(
        make_split(tmp_path, labels), tmp_path, "text_normalized", "image_path", (4, 4),
        class_to_idx={"a": 0, "b": 1}, return_text=True,
    )

    class PredictA(torch.nn.Module):
        def forward(self, images):
            return torch.tensor([[4.0, -4.0]]).repeat(images.size(0), 1)

    metrics = training.evaluate(
        PredictA(), torch.utils.data.DataLoader(dataset, batch_size=2), torch.nn.CrossEntropyLoss(),
        torch.device("cpu"), idx_to_token=dataset.idx_to_class,
    )

    assert metrics["samples"] == 2
    assert metrics["loss_samples"] == expected_samples
    assert metrics["accuracy"] == expected_samples / 2
    assert metrics["cer"] == (2 - expected_samples) / 2
    assert metrics["wer"] == (2 - expected_samples) / 2
    if not expected_samples:
        assert math.isnan(metrics["loss"])


def test_ctc_metrics_include_unknown_and_unalignable_targets(tmp_path: Path) -> None:
    dataset = training.TrainingSplitDataset(
        make_split(tmp_path, ["a", "z", "aa"]), tmp_path, "text_normalized", "image_path", (4, 4),
        char_to_idx={"<blank>": 0, "a": 1}, target_kind=training.CTC_TARGET,
    )

    class PredictA(training.E2CnnBiLSTM):
        def __init__(self):
            # Only the CTC methods are needed; avoid allocating a real backbone.
            torch.nn.Module.__init__(self)
            self.blank_idx = 0
            self.ctc_criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        def forward(self, images):
            return torch.tensor([[[-4.0, 4.0]]]).repeat(images.size(0), 1, 1)

    metrics = training.evaluate(
        PredictA(), torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=training.ctc_collate),
        None, torch.device("cpu"), target_kind=training.CTC_TARGET, idx_to_token=dataset.idx_to_char,
    )

    assert metrics["samples"] == 3
    assert metrics["loss_samples"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["cer"] == pytest.approx(2 / 4)
    assert metrics["wer"] == pytest.approx(2 / 3)
    assert math.isfinite(metrics["loss"])


@pytest.mark.parametrize("samples", [5, 9])
def test_accumulation_matches_batch_gradient_for_uneven_windows(samples: int) -> None:
    torch.manual_seed(7)
    inputs = torch.randn(samples, 3)
    labels = torch.arange(samples) % 2
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    model = torch.nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.07)

    training.train_one_epoch(
        model, torch.utils.data.DataLoader(dataset, batch_size=2), torch.nn.CrossEntropyLoss(),
        optimizer, torch.device("cpu"), gradient_accumulation_steps=3, max_grad_norm=0,
    )
    for values, targets in torch.utils.data.DataLoader(dataset, batch_size=6):
        reference_optimizer.zero_grad()
        torch.nn.functional.cross_entropy(reference(values), targets).backward()
        reference_optimizer.step()

    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_scheduler_counts_partial_accumulation_window() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    args = SimpleNamespace(epochs=3, gradient_accumulation_steps=2, warmup_ratio=0.25)
    scheduler, total, warmup = training.build_scheduler(optimizer, [None] * 3, args)
    assert (total, warmup) == (6, 1)
    assert optimizer.param_groups[0]["lr"] == 0.0
    for _ in range(total):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.0


def test_main_selects_cer_and_records_full_validation_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = make_split(tmp_path, ["a", "b"])
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "train", "--train-csv", str(csv_path), "--val-csv", str(csv_path),
        "--output-dir", str(output), "--epochs", "2", "--num-workers", "0",
        "--image-height", "4", "--image-width", "4", "--no-augmentation",
        "--protocol-version", "equivalent-v2", "--protocol-fingerprint", "fixture",
    ])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "build_model", lambda **kwargs: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 2)))
    monkeypatch.setattr(training, "train_one_epoch", lambda **kwargs: (0.5, 0.5))
    evaluations = iter([
        {"loss": 0.2, "accuracy": 0.8, "cer": 0.5, "wer": 0.5, "samples": 2, "loss_samples": 1},
        {"loss": 0.3, "accuracy": 0.5, "cer": 0.4, "wer": 0.6, "samples": 2, "loss_samples": 1},
    ])
    monkeypatch.setattr(training, "evaluate", lambda **kwargs: next(evaluations))

    training.main()

    config = pd.read_csv(output / "run_config.csv").iloc[0]
    summary = pd.read_csv(output / "run_summary.csv").iloc[0]
    checkpoint = torch.load(output / "e1cnn_best_checkpoint.pt", weights_only=False)
    assert config["effective_batch_size"] == 16
    assert config["validation_population"] == "complete_split"
    assert summary["best_epoch"] == 2
    assert summary["best_val_cer"] == 0.4
    assert summary["final_val_loss_samples"] == 1
    assert checkpoint["best_val_accuracy"] == 0.5
    assert checkpoint["protocol_version"] == "equivalent-v2"
    assert checkpoint["protocol_fingerprint"] == "fixture"
