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
    trocr_processor_metadata,
)


class FakeTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    vocab_size = 8

    def __call__(
        self,
        text: str,
        padding: str | bool,
        max_length: int,
        truncation: bool,
    ) -> SimpleNamespace:
        raw_ids = [3 + (ord(char) % 3) for char in text]
        input_ids = raw_ids[:max_length] if truncation else raw_ids
        if padding == "max_length":
            input_ids += [self.pad_token_id] * max(0, max_length - len(input_ids))
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


@pytest.mark.parametrize("pretrained", [True, False])
def test_load_trocr_components_builds_processor_from_explicit_parts(
    monkeypatch: pytest.MonkeyPatch,
    pretrained: bool,
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

    class FakeModelFactory(FakeVisionEncoderDecoderModel):
        def __init__(self, config=None):
            super().__init__()
            self.used_pretrained_weights = config is None

        @staticmethod
        def from_pretrained(checkpoint: str) -> FakeVisionEncoderDecoderModel:
            del checkpoint
            return FakeModelFactory()

    class FakeConfigFactory:
        @staticmethod
        def from_pretrained(checkpoint: str):
            return SimpleNamespace(source=checkpoint)

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeAutoImageProcessor
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.TrOCRProcessor = FakeTrOCRProcessor
    fake_transformers.VisionEncoderDecoderModel = FakeModelFactory
    fake_transformers.VisionEncoderDecoderConfig = FakeConfigFactory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    args = SimpleNamespace(
        checkpoint="microsoft/trocr-base-handwritten",
        processor_checkpoint=None,
        max_generation_length=16,
        num_beams=2,
        pretrained=pretrained,
    )

    processor, model = load_trocr_components(args)

    assert isinstance(processor, FakeTrOCRProcessor)
    assert isinstance(processor.image_processor, FakeImageProcessor)
    assert processor.tokenizer.pad_token_id == 0
    assert model.config.decoder_start_token_id == 1
    assert model.generation_config.max_length == 16
    assert model.used_pretrained_weights is pretrained


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


def test_trocr_rejects_target_truncation_before_training(tmp_path: Path) -> None:
    csv_path = tmp_path / "val.csv"
    pd.DataFrame(
        [{"image_path": "unused.png", "text_normalized": "long reference"}]
    ).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="sem truncar referencias"):
        TrOCRSplitDataset(
            csv_path, tmp_path, FakeProcessor(), "text_normalized", "image_path",
            max_target_length=4, image_size=(4, 4),
        )


def test_trocr_records_actual_processor_shape() -> None:
    metadata = trocr_processor_metadata(FakeProcessor(), image_size=(64, 512))
    assert metadata["encoder_input_height"] == 4
    assert metadata["encoder_input_width"] == 4
    assert metadata["encoder_input_channels"] == 3


def test_scratch_trocr_initializes_real_encoder_and_decoder_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        transformers.ViTConfig(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, image_size=4, patch_size=2,
        ),
        transformers.TrOCRConfig(
            vocab_size=8, d_model=8, decoder_layers=1, decoder_attention_heads=2,
            decoder_ffn_dim=16, max_position_embeddings=16,
        ),
    )

    def refuse_weights(*args, **kwargs):
        raise AssertionError("Scratch initialization attempted pretrained weights")

    monkeypatch.setattr(transformers.VisionEncoderDecoderModel, "from_pretrained", refuse_weights)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", refuse_weights)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", refuse_weights)
    monkeypatch.setattr(
        transformers.VisionEncoderDecoderConfig, "from_pretrained", lambda *a, **k: config
    )
    monkeypatch.setattr(
        transformers.TrOCRProcessor, "from_pretrained", lambda *a, **k: FakeProcessor()
    )
    _, model = load_trocr_components(SimpleNamespace(
        checkpoint="architecture-only", processor_checkpoint=None, pretrained=False,
        max_generation_length=8, num_beams=1,
    ))
    assert isinstance(model.encoder, transformers.ViTModel)
    assert isinstance(model.decoder, transformers.TrOCRForCausalLM)
    assert model.encoder.embeddings.patch_embeddings.projection.weight.std() > 0


@pytest.mark.parametrize("model_kind", ["trocr", "parseq"])
@pytest.mark.parametrize("sample_count", [5, 9, 12])
def test_accumulation_matches_actual_window_sample_mean(
    monkeypatch: pytest.MonkeyPatch, model_kind: str, sample_count: int,
) -> None:
    import importlib

    module = importlib.import_module(
        f"med_ocr_research.training.train_{model_kind}"
    )
    from med_ocr_research.training.train_trocr import build_grad_scaler

    class TinyRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, pixel_values, labels):
            return SimpleNamespace(
                loss=((pixel_values * self.weight - labels) ** 2).mean()
            )

    model = TinyRegressor()
    reference = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    x = torch.linspace(0.1, 1.0, sample_count)
    y = torch.linspace(0.7, 0.2, sample_count)
    batches = [
        {
            "pixel_values": x[start:start + 2],
            "images": x[start:start + 2],
            "labels": y[start:start + 2],
            "texts": y[start:start + 2],
        }
        for start in range(0, sample_count, 2)
    ]
    if model_kind == "parseq":
        monkeypatch.setattr(
            module, "parseq_permutation_loss",
            lambda model, images, texts: model(images, texts).loss,
        )

    module.train_one_epoch(
        model, batches, optimizer, scheduler, build_grad_scaler(False),
        torch.device("cpu"),
        SimpleNamespace(fp16=False, gradient_accumulation_steps=3, max_grad_norm=0.2),
    )
    for start in range(0, sample_count, 6):
        reference_optimizer.zero_grad()
        reference(x[start:start + 6], y[start:start + 6]).loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.2)
        reference_optimizer.step()

    torch.testing.assert_close(model.weight, reference.weight)
    assert scheduler.last_epoch == (sample_count + 5) // 6


@pytest.mark.parametrize("model_kind", ["trocr", "parseq"])
def test_validation_keeps_every_reference_and_empty_prediction(
    monkeypatch: pytest.MonkeyPatch, model_kind: str,
) -> None:
    import importlib

    module = importlib.import_module(
        f"med_ocr_research.training.train_{model_kind}"
    )

    class ValidationModel:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            if "labels" in kwargs:
                return SimpleNamespace(loss=torch.tensor(1.0))
            return args[0]

        def generate(self, images, **kwargs):
            return images

        def forward_logits_loss(self, images, texts):
            return None, torch.tensor(1.0), None

    def decode(ids, **kwargs):
        return [["known", "", "wrong"][int(i)] for i in ids]

    batches = [
        {"pixel_values": torch.tensor([0, 1]), "images": torch.tensor([0, 1]),
         "labels": torch.tensor([0, 1]), "texts": ["known", "unseen phrase"]},
        {"pixel_values": torch.tensor([2]), "images": torch.tensor([2]),
         "labels": torch.tensor([2]), "texts": ["last"]},
    ]
    args = SimpleNamespace(max_generation_length=64, num_beams=1, batch_size=99)
    if model_kind == "trocr":
        _, metrics, rows = module.evaluate(
            ValidationModel(), batches, SimpleNamespace(batch_decode=decode),
            torch.device("cpu"), args,
        )
    else:
        monkeypatch.setattr(
            module, "decode_parseq_logits", lambda model, logits: (decode(logits), [])
        )
        _, metrics, rows = module.evaluate(
            ValidationModel(), batches, torch.device("cpu"), args,
        )
    assert metrics["samples"] == 3
    assert metrics["exact_match_accuracy"] == pytest.approx(1 / 3)
    assert rows["row_index"].tolist() == [0, 1, 2]
    assert rows["text_true"].tolist() == ["known", "unseen phrase", "last"]
    assert rows["text_pred"].tolist() == ["known", "", "wrong"]
