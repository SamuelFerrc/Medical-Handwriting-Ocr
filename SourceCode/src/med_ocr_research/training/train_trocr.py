from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from torchvision import transforms
except ImportError:
    transforms = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.evaluation.metrics import aggregate_metrics
from med_ocr_research.training.augmentation import build_handwriting_train_transform


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs/runs"
DEFAULT_SCENARIOS_DIR = PROJECT_ROOT / "data/splits/scenarios"
DEFAULT_DATASET = "prescriptionbd"
DEFAULT_CHECKPOINT = "microsoft/trocr-base-handwritten"
DEFAULT_RUN_NAME_PREFIX = "trocr"
SUPPORTED_DATASETS = ("prescriptionbd", "rxhandbd", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Faz fine-tuning do TrOCR em um split selecionavel. Use --dataset "
            "para os cenarios prontos ou --train-csv/--val-csv para outro dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default=DEFAULT_DATASET,
        help=(
            "Cenario usado quando --train-csv/--val-csv nao sao informados. "
            "Opcoes seguem data/splits/scenarios/."
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="CSV de treino. Se omitido, usa data/splits/scenarios/<dataset>/train.csv.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="CSV de validacao. Se omitido, usa data/splits/scenarios/<dataset>/val.csv.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sobrescreve outputs/runs/<run-name> quando informado.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint Hugging Face do TrOCR usado como ponto de partida.",
    )
    parser.add_argument(
        "--processor-checkpoint",
        default=None,
        help=(
            "Checkpoint do processor/tokenizer. Por padrao usa o mesmo valor de "
            "--checkpoint."
        ),
    )
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Micro-batch por passo. Use gradient accumulation para manter "
            "batch efetivo maior sem estourar VRAM."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-generation-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Com batch-size 2, o padrao mantem batch efetivo 16.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Reduz uso de VRAM recomputando ativacoes durante o backward.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Congela o encoder visual e treina apenas o decoder/cabecas.",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="medium",
    )
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_run_name(run_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("O nome da execucao nao pode ficar vazio.")
    return normalized


def default_run_name(dataset: str | None, checkpoint: str) -> str:
    checkpoint_name = checkpoint.rstrip("/").split("/")[-1] or DEFAULT_RUN_NAME_PREFIX
    parts = [DEFAULT_RUN_NAME_PREFIX, checkpoint_name]
    if dataset:
        parts.append(dataset)
    return sanitize_run_name("_".join(parts))


def resolve_output_dir(run_name: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return DEFAULT_RUNS_DIR / sanitize_run_name(run_name)


def resolve_split_paths(
    dataset: str,
    train_csv: Path | None,
    val_csv: Path | None,
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
) -> tuple[Path, Path]:
    if train_csv is not None and val_csv is not None:
        return train_csv, val_csv

    split_dir = scenarios_dir / dataset
    resolved_train_csv = train_csv or split_dir / "train.csv"
    resolved_val_csv = val_csv or split_dir / "val.csv"
    missing = [
        path
        for path in (resolved_train_csv, resolved_val_csv)
        if not Path(path).exists()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Split ausente para dataset {dataset!r}: {missing_paths}"
        )
    return resolved_train_csv, resolved_val_csv


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(str(image_path))
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    return data_root / path


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trocr_components(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers nao esta instalado. Rode: "
            "python -m pip install -r requirements-train.txt"
        ) from exc

    processor_checkpoint = args.processor_checkpoint or args.checkpoint
    try:
        processor = TrOCRProcessor.from_pretrained(processor_checkpoint)
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained(processor_checkpoint)
        try:
            tokenizer = AutoTokenizer.from_pretrained(processor_checkpoint)
        except Exception as exc:
            raise RuntimeError(
                "Nao foi possivel carregar o tokenizer do TrOCR. Confira se "
                "sentencepiece e protobuf estao instalados com "
                "python -m pip install -r requirements-train.txt."
            ) from exc
        try:
            processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
        except TypeError:
            processor = TrOCRProcessor(
                feature_extractor=image_processor,
                tokenizer=tokenizer,
            )
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    configure_generation(model, processor, args)
    return processor, model


def configure_generation(model: Any, processor: Any, args: argparse.Namespace) -> None:
    tokenizer = processor.tokenizer
    decoder_start_token_id = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )
    eos_token_id = (
        tokenizer.sep_token_id
        if tokenizer.sep_token_id is not None
        else tokenizer.eos_token_id
    )
    if decoder_start_token_id is None:
        raise ValueError("Tokenizer sem cls_token_id/bos_token_id para iniciar o decoder.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer sem pad_token_id para mascarar os labels.")

    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if hasattr(model, "generation_config"):
        model.generation_config.decoder_start_token_id = decoder_start_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = eos_token_id
        model.generation_config.max_length = args.max_generation_length
        model.generation_config.num_beams = args.num_beams


def freeze_encoder_if_requested(model: Any, freeze_encoder: bool) -> None:
    if not freeze_encoder:
        return
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False


def enable_gradient_checkpointing_if_requested(
    model: Any,
    gradient_checkpointing: bool,
) -> None:
    if not gradient_checkpointing:
        return
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError("Este modelo nao suporta gradient checkpointing.")

    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "use_cache"):
        model.config.decoder.use_cache = False
    if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
        model.decoder.config.use_cache = False


def build_train_transform(strength: str) -> transforms.Compose:
    return build_handwriting_train_transform(transforms, strength)


class TrOCRSplitDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        processor: Any,
        label_column: str,
        image_column: str,
        max_target_length: int,
        image_size: tuple[int, int],
        split: str | None = None,
        image_transform: transforms.Compose | None = None,
        limit: int | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.processor = processor
        self.label_column = label_column
        self.image_column = image_column
        self.max_target_length = max_target_length
        self.image_size = image_size
        self.image_transform = image_transform

        rows = pd.read_csv(csv_path)
        if split is not None and "split" in rows.columns:
            rows = rows[rows["split"].astype(str).str.lower() == split].copy()

        required = {image_column, label_column}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

        rows = rows.dropna(subset=[image_column, label_column]).copy()
        rows[label_column] = rows[label_column].astype(str)
        rows = rows[rows[label_column].str.len() > 0].copy()
        if limit is not None:
            rows = rows.head(limit).copy()
        if rows.empty:
            raise ValueError(f"Nenhuma amostra encontrada em {csv_path}")

        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image_path = resolve_image_path(self.data_root, row[self.image_column])
        text = str(row[self.label_column])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size[1], self.image_size[0]))
            if self.image_transform is not None:
                image = self.image_transform(image)
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)

        tokenized = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        )
        labels = [
            token_id
            if token_id != self.processor.tokenizer.pad_token_id
            else -100
            for token_id in tokenized.input_ids
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text": text,
        }


def trocr_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }


def decode_label_batch(labels: torch.Tensor, processor: Any) -> list[str]:
    pad_token_id = processor.tokenizer.pad_token_id
    label_ids = labels.detach().cpu().clone()
    label_ids[label_ids == -100] = pad_token_id
    return processor.batch_decode(label_ids, skip_special_tokens=True)


def save_text_distribution_csv(
    train_dataset: TrOCRSplitDataset,
    val_dataset: TrOCRSplitDataset,
    output_path: Path,
) -> None:
    rows = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        counts = dataset.rows[dataset.label_column].value_counts().to_dict()
        total = len(dataset)
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "fraction": int(count) / total if total else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_character_distribution_csv(
    train_dataset: TrOCRSplitDataset,
    val_dataset: TrOCRSplitDataset,
    output_path: Path,
) -> None:
    rows = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        counts: dict[str, int] = {}
        for label in dataset.rows[dataset.label_column]:
            for char in str(label):
                counts[char] = counts.get(char, 0) + 1
        total = sum(counts.values())
        for char, count in sorted(counts.items()):
            rows.append(
                {
                    "split": split_name,
                    "character": char,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_run_config_csv(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    train_csv: Path,
    val_csv: Path,
    train_dataset: TrOCRSplitDataset,
    val_dataset: TrOCRSplitDataset,
    processor: Any,
    output_path: Path,
) -> None:
    tokenizer = processor.tokenizer
    pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "model_kind": "trocr",
                "checkpoint": args.checkpoint,
                "processor_checkpoint": args.processor_checkpoint or args.checkpoint,
                "dataset": args.dataset,
                "output_dir": str(output_dir),
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "data_root": str(args.data_root),
                "label_column": args.label_column,
                "image_column": args.image_column,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
                "max_target_length": args.max_target_length,
                "max_generation_length": args.max_generation_length,
                "num_beams": args.num_beams,
                "image_height": args.image_height,
                "image_width": args.image_width,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_ratio": args.warmup_ratio,
                "max_grad_norm": args.max_grad_norm,
                "fp16": args.fp16,
                "gradient_checkpointing": args.gradient_checkpointing,
                "freeze_encoder": args.freeze_encoder,
                "seed": args.seed,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else ""
                ),
                "augmentation_enabled": not args.no_augmentation,
                "augmentation_strength": (
                    "" if args.no_augmentation else args.augmentation_strength
                ),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", ""),
                "pad_token_id": tokenizer.pad_token_id,
                "decoder_start_token_id": getattr(
                    processor.tokenizer,
                    "cls_token_id",
                    None,
                ),
            }
        ]
    ).to_csv(output_path, index=False)


def optimizer_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
) -> None:
    if args.max_grad_norm and args.max_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scale_before_step = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after_step = scaler.get_scale()
    if (not scaler.is_enabled()) or scale_after_step >= scale_before_step:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def build_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    use_amp = bool(args.fp16 and device.type == "cuda")
    progress = tqdm(dataloader, desc="Treino", leave=False)

    for batch_index, batch in enumerate(progress, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        batch_size = pixel_values.size(0)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )

        with autocast_context:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()
        should_step = (
            batch_index % args.gradient_accumulation_steps == 0
            or batch_index == len(dataloader)
        )
        if should_step:
            optimizer_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
            )

        loss_value = float(loss.detach().item()) * args.gradient_accumulation_steps
        total_loss += loss_value * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / total_samples


def evaluate(
    model: Any,
    dataloader: DataLoader,
    processor: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | int], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    references: list[str] = []
    predictions: list[str] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(dataloader, desc="Validacao", leave=False)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            generated_ids = model.generate(
                pixel_values,
                max_length=args.max_generation_length,
                num_beams=args.num_beams,
            )
            batch_predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            batch_references = batch["texts"]

            batch_size = pixel_values.size(0)
            total_loss += float(outputs.loss.detach().item()) * batch_size
            total_samples += batch_size
            references.extend(batch_references)
            predictions.extend(batch_predictions)

            start_index = batch_index * args.batch_size
            for offset, (reference, prediction) in enumerate(
                zip(batch_references, batch_predictions)
            ):
                prediction_rows.append(
                    {
                        "row_index": start_index + offset,
                        "text_true": reference,
                        "text_pred": prediction,
                    }
                )

    metrics = aggregate_metrics(references, predictions)
    metrics_dict: dict[str, float | int] = {
        "samples": metrics.samples,
        "exact_match_accuracy": metrics.exact_match_accuracy,
        "cer": metrics.cer,
        "wer": metrics.wer,
        "correct": metrics.correct,
        "total_char_errors": metrics.total_char_errors,
        "total_reference_chars": metrics.total_reference_chars,
        "total_word_errors": metrics.total_word_errors,
        "total_reference_words": metrics.total_reference_words,
    }
    average_loss = total_loss / total_samples if total_samples else 0.0
    return average_loss, metrics_dict, pd.DataFrame(prediction_rows)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    args: argparse.Namespace,
) -> tuple[Any, int, int]:
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError(
            "transformers nao esta instalado. Rode: "
            "python -m pip install -r requirements-train.txt"
        ) from exc

    update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_steps = max(1, update_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return scheduler, total_steps, warmup_steps


def save_model_bundle(
    model: Any,
    processor: Any,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train_csv, val_csv = resolve_split_paths(
        dataset=args.dataset,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
    )
    if args.run_name is None:
        args.run_name = default_run_name(args.dataset, args.checkpoint)
    args.run_name = sanitize_run_name(args.run_name)

    seed_everything(args.seed)
    device = choose_device()
    output_dir = resolve_output_dir(args.run_name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_trocr_components(args)
    freeze_encoder_if_requested(model, args.freeze_encoder)
    enable_gradient_checkpointing_if_requested(model, args.gradient_checkpointing)
    model.to(device)

    train_transform = (
        None
        if args.no_augmentation
        else build_train_transform(args.augmentation_strength)
    )
    train_dataset = TrOCRSplitDataset(
        csv_path=train_csv,
        data_root=args.data_root,
        processor=processor,
        label_column=args.label_column,
        image_column=args.image_column,
        max_target_length=args.max_target_length,
        image_size=(args.image_height, args.image_width),
        split="train",
        image_transform=train_transform,
        limit=args.limit_train_samples,
    )
    val_dataset = TrOCRSplitDataset(
        csv_path=val_csv,
        data_root=args.data_root,
        processor=processor,
        label_column=args.label_column,
        image_column=args.image_column,
        max_target_length=args.max_target_length,
        image_size=(args.image_height, args.image_width),
        split="val",
        limit=args.limit_val_samples,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=trocr_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=trocr_collate,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler, total_steps, warmup_steps = build_scheduler(
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        args=args,
    )
    scaler = build_grad_scaler(enabled=bool(args.fp16 and device.type == "cuda"))

    history_csv_path = output_dir / "training_history.csv"
    run_config_csv_path = output_dir / "run_config.csv"
    run_summary_csv_path = output_dir / "run_summary.csv"
    text_distribution_csv_path = output_dir / "text_distribution.csv"
    character_distribution_csv_path = output_dir / "character_distribution.csv"
    best_predictions_csv_path = output_dir / "best_val_predictions.csv"
    final_model_dir = output_dir / "trocr_final"
    best_model_dir = output_dir / "trocr_best"

    save_text_distribution_csv(train_dataset, val_dataset, text_distribution_csv_path)
    save_character_distribution_csv(
        train_dataset,
        val_dataset,
        character_distribution_csv_path,
    )
    save_run_config_csv(
        args=args,
        output_dir=output_dir,
        device=device,
        train_csv=train_csv,
        val_csv=val_csv,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor,
        output_path=run_config_csv_path,
    )

    print(f"Nome da execucao: {args.run_name}")
    print(f"Modelo base: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Pasta da execucao: {output_dir}")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CSV de treino: {train_csv}")
    print(f"CSV de validacao: {val_csv}")
    print(f"Amostras de treino: {len(train_dataset)}")
    print(f"Amostras de validacao: {len(val_dataset)}")
    print(f"Passos de otimizacao: {total_steps} (warmup: {warmup_steps})")
    print(f"Data augmentation: {'desligado' if args.no_augmentation else 'ligado'}")
    if args.freeze_encoder:
        print("Encoder visual congelado.")
    if args.gradient_checkpointing:
        print("Gradient checkpointing: ligado.")

    best_val_cer = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_val_loss = None
    history_rows: list[dict[str, Any]] = []
    best_predictions = pd.DataFrame()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
        )
        val_loss, val_metrics, val_predictions = evaluate(
            model=model,
            dataloader=val_dataloader,
            processor=processor,
            device=device,
            args=args,
        )
        val_accuracy = float(val_metrics["exact_match_accuracy"])
        val_cer = float(val_metrics["cer"])
        val_wer = float(val_metrics["wer"])
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"- loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_accuracy: {val_accuracy:.4f} "
            f"- val_cer: {val_cer:.4f} "
            f"- val_wer: {val_wer:.4f}"
        )

        is_best_epoch = (val_cer < best_val_cer) or (
            val_cer == best_val_cer and val_accuracy > best_val_accuracy
        )
        if is_best_epoch:
            best_val_cer = val_cer
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            best_predictions = val_predictions.copy()
            metadata = {
                "run_name": args.run_name,
                "model_kind": "trocr",
                "checkpoint": args.checkpoint,
                "dataset": args.dataset,
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "label_column": args.label_column,
                "image_column": args.image_column,
                "image_size": [args.image_height, args.image_width],
                "max_target_length": args.max_target_length,
                "max_generation_length": args.max_generation_length,
                "num_beams": args.num_beams,
                "gradient_checkpointing": args.gradient_checkpointing,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "best_val_cer": best_val_cer,
                "best_val_wer": val_wer,
            }
            save_model_bundle(model, processor, best_model_dir, metadata)
            best_predictions.to_csv(best_predictions_csv_path, index=False)
            print(
                f"Novo melhor TrOCR salvo: epoch {best_epoch} "
                f"- val_cer: {best_val_cer:.4f}"
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "learning_rate": scheduler.get_last_lr()[0],
                "is_best_epoch": is_best_epoch,
                "best_epoch_so_far": best_epoch,
                "best_val_accuracy_so_far": best_val_accuracy,
                "best_val_cer_so_far": best_val_cer,
                "best_val_loss_so_far": best_val_loss,
            }
        )
        pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)

    final_metadata = {
        "run_name": args.run_name,
        "model_kind": "trocr",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "label_column": args.label_column,
        "image_column": args.image_column,
        "image_size": [args.image_height, args.image_width],
        "max_target_length": args.max_target_length,
        "max_generation_length": args.max_generation_length,
        "num_beams": args.num_beams,
        "gradient_checkpointing": args.gradient_checkpointing,
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_val_cer": best_val_cer,
    }
    save_model_bundle(model, processor, final_model_dir, final_metadata)

    if history_rows:
        final_metrics = history_rows[-1]
        pd.DataFrame(
            [
                {
                    "run_name": args.run_name,
                    "output_dir": str(output_dir),
                    "epochs_completed": len(history_rows),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_cer": best_val_cer,
                    "best_val_loss": best_val_loss,
                    "final_train_loss": final_metrics["train_loss"],
                    "final_val_loss": final_metrics["val_loss"],
                    "final_val_accuracy": final_metrics["val_accuracy"],
                    "final_val_cer": final_metrics["val_cer"],
                    "final_val_wer": final_metrics["val_wer"],
                    "final_model_dir": str(final_model_dir),
                    "best_model_dir": str(best_model_dir),
                    "history_csv_path": str(history_csv_path),
                    "best_predictions_csv_path": str(best_predictions_csv_path),
                    "text_distribution_csv_path": str(text_distribution_csv_path),
                    "character_distribution_csv_path": str(
                        character_distribution_csv_path
                    ),
                    "run_config_csv_path": str(run_config_csv_path),
                }
            ]
        ).to_csv(run_summary_csv_path, index=False)

    print(f"Modelo final salvo em: {final_model_dir}")
    print(
        f"Melhor modelo salvo em: {best_model_dir} "
        f"(epoch {best_epoch}, val_cer: {best_val_cer:.4f})"
    )
    print(f"Historico de treino salvo em: {history_csv_path}")
    print(f"Resumo da execucao salvo em: {run_summary_csv_path}")
    print(f"Predicoes da melhor validacao salvas em: {best_predictions_csv_path}")
    print(f"Configuracao da execucao salva em: {run_config_csv_path}")


if __name__ == "__main__":
    main()
