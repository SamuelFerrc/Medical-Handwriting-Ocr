from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from med_ocr_research.training.train_trocr import (
    TrOCRSplitDataset,
    load_trocr_components,
    resolve_split_paths,
    sanitize_run_name,
    trocr_collate,
)


class FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    vocab_size = 8

    def __call__(
        self,
        text: str,
        padding: str,
        max_length: int,
        truncation: bool,
    ) -> SimpleNamespace:
        del padding, truncation
        raw_ids = [3 + (ord(char) % 3) for char in text]
        input_ids = (raw_ids + [self.pad_token_id] * max_length)[:max_length]
        return SimpleNamespace(input_ids=input_ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, images: Image.Image, return_tensors: str) -> SimpleNamespace:
        del images, return_tensors
        return SimpleNamespace(pixel_values=torch.ones(1, 3, 4, 4))

    def batch_decode(self, ids, skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        return ["decoded" for _ in ids]


class FakeImageProcessor:
    pass


class FakeTrOCRProcessor(FakeProcessor):
    @classmethod
    def from_pretrained(cls, checkpoint: str) -> "FakeTrOCRProcessor":
        del checkpoint
        raise OSError("processor config unavailable")

    def __init__(self, image_processor: FakeImageProcessor, tokenizer: FakeTokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer


class FakeVisionEncoderDecoderModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(decoder=SimpleNamespace(vocab_size=8))
        self.generation_config = SimpleNamespace()


def test_load_trocr_components_builds_processor_from_explicit_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(checkpoint: str) -> FakeImageProcessor:
            del checkpoint
            return FakeImageProcessor()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(checkpoint: str) -> FakeTokenizer:
            del checkpoint
            return FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(checkpoint: str) -> FakeVisionEncoderDecoderModel:
            del checkpoint
            return FakeVisionEncoderDecoderModel()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.TrOCRProcessor = FakeTrOCRProcessor
    fake_transformers.VisionEncoderDecoderModel = FakeModelFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    args = SimpleNamespace(
        checkpoint="microsoft/trocr-base-handwritten",
        processor_checkpoint=None,
        max_generation_length=16,
        num_beams=2,
    )

    processor, model = load_trocr_components(args)

    assert isinstance(processor, FakeTrOCRProcessor)
    assert isinstance(processor.image_processor, FakeImageProcessor)
    assert processor.tokenizer.pad_token_id == 0
    assert model.config.decoder_start_token_id == 1
    assert model.generation_config.max_length == 16


def test_resolve_split_paths_uses_selected_dataset(tmp_path: Path) -> None:
    split_dir = tmp_path / "rxhandbd"
    split_dir.mkdir()
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    train_csv.write_text("image_path,text_normalized\n", encoding="utf-8")
    val_csv.write_text("image_path,text_normalized\n", encoding="utf-8")

    resolved_train, resolved_val = resolve_split_paths(
        dataset="rxhandbd",
        train_csv=None,
        val_csv=None,
        scenarios_dir=tmp_path,
    )

    assert resolved_train == train_csv
    assert resolved_val == val_csv


def test_sanitize_run_name_rejects_empty_names() -> None:
    with pytest.raises(ValueError):
        sanitize_run_name(" / ")


def test_trocr_dataset_builds_pixel_values_and_masked_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "text_normalized": "abc",
                "split": "train",
            }
        ]
    ).to_csv(csv_path, index=False)

    dataset = TrOCRSplitDataset(
        csv_path=csv_path,
        data_root=tmp_path,
        processor=FakeProcessor(),
        label_column="text_normalized",
        image_column="image_path",
        max_target_length=6,
        image_size=(4, 4),
        split="train",
    )

    item = dataset[0]

    assert item["pixel_values"].shape == (3, 4, 4)
    assert item["labels"].shape == (6,)
    assert item["labels"][-1].item() == -100
    assert item["text"] == "abc"


def test_trocr_collate_stacks_batch(tmp_path: Path) -> None:
    del tmp_path
    batch = [
        {
            "pixel_values": torch.ones(3, 4, 4),
            "labels": torch.tensor([1, -100]),
            "text": "a",
        },
        {
            "pixel_values": torch.zeros(3, 4, 4),
            "labels": torch.tensor([2, -100]),
            "text": "b",
        },
    ]

    collated = trocr_collate(batch)

    assert collated["pixel_values"].shape == (2, 3, 4, 4)
    assert collated["labels"].shape == (2, 2)
    assert collated["texts"] == ["a", "b"]
