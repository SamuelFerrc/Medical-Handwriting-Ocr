from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from med_ocr_research.models.E2CnnBiLSTM import E2CnnBiLSTM


def test_e2cnn_bilstm_returns_temporal_logits_by_default() -> None:
    model = E2CnnBiLSTM(num_classes=6)
    images = torch.randn(2, 1, 64, 128)

    logits = model(images)

    assert logits.shape == (2, 16, 6)


def test_e2cnn_bilstm_can_return_legacy_classifier_logits() -> None:
    model = E2CnnBiLSTM(num_classes=6, default_return_sequence=False)
    images = torch.randn(2, 1, 64, 128)

    logits = model(images)

    assert logits.shape == (2, 6)


def test_ctc_loss_returns_finite_scalar() -> None:
    model = E2CnnBiLSTM(num_classes=4)
    logits = torch.randn(2, 8, 4, requires_grad=True)
    targets = torch.tensor([1, 2, 1], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    loss = model.ctc_loss(logits, targets, target_lengths)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ctc_decode_collapses_repeats_and_removes_blank() -> None:
    model = E2CnnBiLSTM(num_classes=4, blank_idx=0)
    idx_to_char = {0: "<blank>", 1: "a", 2: "b", 3: "c"}
    raw_indices = [0, 1, 1, 0, 1, 2, 2, 0, 3]
    logits = torch.full((1, len(raw_indices), 4), -10.0)
    for timestep, index in enumerate(raw_indices):
        logits[0, timestep, index] = 10.0

    decoded = model.ctc_decode(logits, idx_to_token=idx_to_char)

    assert decoded == ["aabc"]
