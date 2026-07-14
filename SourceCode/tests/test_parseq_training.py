from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from med_ocr_research.models.parseq_utils import (
    adapt_parseq_charset,
    adapt_parseq_dimensions,
    tokenizer_charset,
    validate_parseq_labels,
)
from med_ocr_research.training.train_parseq import (
    PARSeqSplitDataset,
    parseq_collate,
    sanitize_run_name,
)


class FakeTokenizer:
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str) -> None:
        self._itos = (self.EOS,) + tuple(charset) + (self.BOS, self.PAD)
        self._stoi = {token: index for index, token in enumerate(self._itos)}
        self.eos_id = self._stoi[self.EOS]
        self.bos_id = self._stoi[self.BOS]
        self.pad_id = self._stoi[self.PAD]

    def __len__(self) -> int:
        return len(self._itos)


def fake_parseq(charset: str = "ab") -> SimpleNamespace:
    tokenizer = FakeTokenizer(charset)
    embedding = torch.nn.Embedding(len(tokenizer), 4)
    head = torch.nn.Linear(4, len(tokenizer) - 2)
    with torch.no_grad():
        embedding.weight.copy_(
            torch.arange(len(tokenizer) * 4, dtype=torch.float32).reshape(-1, 4)
        )
        head.weight.copy_(
            torch.arange(head.out_features * 4, dtype=torch.float32).reshape(-1, 4)
        )
    return SimpleNamespace(
        tokenizer=tokenizer,
        model=SimpleNamespace(
            text_embed=SimpleNamespace(embedding=embedding),
            head=head,
        ),
        hparams=SimpleNamespace(charset_train=charset, max_label_length=25),
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
    )


def test_parseq_dataset_applies_expected_shape_and_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "text_normalized": "500 mg",
                "split": "train",
            }
        ]
    ).to_csv(csv_path, index=False)

    dataset = PARSeqSplitDataset(
        csv_path=csv_path,
        data_root=tmp_path,
        label_column="text_normalized",
        image_column="image_path",
        image_size=(32, 128),
        split="train",
    )
    item = dataset[0]

    assert item["images"].shape == (3, 32, 128)
    assert torch.allclose(item["images"], torch.ones_like(item["images"]))
    assert item["text"] == "500 mg"


def test_parseq_collate_stacks_images_and_keeps_texts() -> None:
    batch = [
        {"images": torch.ones(3, 2, 4), "text": "a"},
        {"images": torch.zeros(3, 2, 4), "text": "b"},
    ]
    collated = parseq_collate(batch)
    assert collated["images"].shape == (2, 3, 2, 4)
    assert collated["texts"] == ["a", "b"]


def test_adapt_parseq_charset_preserves_pretrained_rows() -> None:
    model = fake_parseq()
    old_eos = model.model.head.weight[model.eos_id].detach().clone()
    old_a = model.model.head.weight[model.tokenizer._stoi["a"]].detach().clone()

    adapt_parseq_charset(model, "ab ")

    assert tokenizer_charset(model.tokenizer) == "ab "
    assert model.model.head.out_features == 4
    assert torch.equal(model.model.head.weight[model.eos_id], old_eos)
    assert torch.equal(model.model.head.weight[model.tokenizer._stoi["a"]], old_a)
    assert model.tokenizer._stoi[" "] == 3


def test_validate_parseq_labels_reports_unsupported_characters() -> None:
    model = fake_parseq("ab ")
    validate_parseq_labels(["a b"], model, 25, "train")
    with pytest.raises(ValueError, match="'x'"):
        validate_parseq_labels(["ax"], model, 25, "train")


def test_adapt_parseq_dimensions_interpolates_visual_and_text_positions() -> None:
    patch_embed = SimpleNamespace(
        img_size=(4, 8),
        grid_size=(2, 4),
        patch_size=(2, 2),
        num_patches=8,
    )
    pos_queries = torch.nn.Parameter(torch.randn(1, 3, 4))
    old_queries = pos_queries.detach().clone()
    model = SimpleNamespace(
        model=SimpleNamespace(
            encoder=SimpleNamespace(
                patch_embed=patch_embed,
                pos_embed=torch.nn.Parameter(torch.randn(1, 8, 4)),
            ),
            pos_queries=pos_queries,
            max_label_length=2,
        ),
        hparams=SimpleNamespace(img_size=[4, 8], max_label_length=2),
    )

    adapt_parseq_dimensions(model, (8, 16), 6)

    assert model.model.encoder.pos_embed.shape == (1, 32, 4)
    assert model.model.encoder.patch_embed.grid_size == (4, 8)
    assert model.model.pos_queries.shape == (1, 7, 4)
    assert torch.equal(model.model.pos_queries[:, :3], old_queries)
    assert model.hparams.img_size == [8, 16]
    assert model.hparams.max_label_length == 6


def test_sanitize_run_name_rejects_empty_names() -> None:
    with pytest.raises(ValueError):
        sanitize_run_name(" / ")
