from __future__ import annotations

import argparse
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
from med_ocr_research.models.parseq_utils import (
    DEFAULT_PARSEQ_HUB_REPO,
    DEFAULT_PARSEQ_MODEL,
    SUPPORTED_PARSEQ_MODELS,
    adapt_parseq_charset,
    adapt_parseq_dimensions,
    charset_with_extras,
    configure_parseq_gradient_checkpointing,
    decode_parseq_logits,
    freeze_parseq_encoder,
    load_official_parseq,
    parseq_input_size,
    parseq_max_label_length,
    parseq_permutation_loss,
    save_parseq_bundle,
    tokenizer_charset,
    validate_parseq_labels,
)
from med_ocr_research.training.augmentation import build_handwriting_train_transform
from med_ocr_research.training.train_trocr import (
    build_grad_scaler,
    build_scheduler,
    choose_device,
    optimizer_step,
    resolve_image_path,
    resolve_split_paths,
    save_character_distribution_csv,
    save_text_distribution_csv,
    seed_everything,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs/runs"
DEFAULT_DATASET = "prescriptionbd"
DEFAULT_RUN_NAME_PREFIX = "parseq"
SUPPORTED_DATASETS = ("prescriptionbd", "rxhandbd", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Faz fine-tuning do PARSeq oficial nos mesmos splits e com os mesmos "
            "artefatos usados pelo TrOCR."
        )
    )
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model-name",
        choices=SUPPORTED_PARSEQ_MODELS,
        default=DEFAULT_PARSEQ_MODEL,
        help="Entrada do Torch Hub oficial do PARSeq.",
    )
    parser.add_argument("--hub-repo", default=DEFAULT_PARSEQ_HUB_REPO)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--extra-characters",
        default=" ",
        help="Caracteres ausentes no vocabulario oficial; espaco e incluido por padrao.",
    )
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-generation-length", type=int, default=64)
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Parametro de compatibilidade; PARSeq usa decodificacao greedy.",
    )
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="strong",
    )
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sanitize_run_name(run_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("O nome da execucao nao pode ficar vazio.")
    return normalized


def default_run_name(dataset: str, model_name: str) -> str:
    return sanitize_run_name(f"{DEFAULT_RUN_NAME_PREFIX}_{model_name}_{dataset}")


def resolve_output_dir(run_name: str, output_dir: Path | None) -> Path:
    return output_dir if output_dir is not None else DEFAULT_RUNS_DIR / run_name


def build_train_transform(strength: str) -> Any:
    return build_handwriting_train_transform(transforms, strength)


def build_image_transform() -> Any:
    if transforms is None:
        raise RuntimeError(
            "torchvision nao esta instalado. Rode: python -m pip install -r "
            "requirements-train.txt"
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )


class PARSeqSplitDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        label_column: str,
        image_column: str,
        image_size: tuple[int, int],
        split: str | None = None,
        image_transform: Any | None = None,
        limit: int | None = None,
    ) -> None:
        self.csv_path = csv_path
        self.data_root = data_root
        self.label_column = label_column
        self.image_column = image_column
        self.image_size = image_size
        self.image_transform = image_transform
        self.tensor_transform = build_image_transform()

        rows = pd.read_csv(csv_path)
        if split is not None and "split" in rows.columns:
            rows = rows[rows["split"].astype(str).str.lower() == split].copy()
        missing = {image_column, label_column}.difference(rows.columns)
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
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size[1], self.image_size[0]))
            if self.image_transform is not None:
                image = self.image_transform(image)
            image_tensor = self.tensor_transform(image)
        return {"images": image_tensor, "text": str(row[self.label_column])}


def parseq_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }


def save_run_config_csv(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    train_csv: Path,
    val_csv: Path,
    train_dataset: PARSeqSplitDataset,
    val_dataset: PARSeqSplitDataset,
    model: Any,
    output_path: Path,
) -> None:
    pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "model_kind": "parseq",
                "model_name": args.model_name,
                "hub_repo": args.hub_repo,
                "pretrained": args.pretrained,
                "dataset": args.dataset,
                "output_dir": str(output_dir),
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "data_root": str(args.data_root),
                "label_column": args.label_column,
                "image_column": args.image_column,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "effective_batch_size": (
                    args.batch_size * args.gradient_accumulation_steps
                ),
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
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                ),
                "augmentation_enabled": not args.no_augmentation,
                "augmentation_strength": (
                    "" if args.no_augmentation else args.augmentation_strength
                ),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "charset": tokenizer_charset(model.tokenizer),
                "vocab_size": len(model.tokenizer),
            }
        ]
    ).to_csv(output_path, index=False)


def train_one_epoch(
    model: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
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
        images = batch["images"].to(device)
        texts = batch["texts"]
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast_context:
            batch_loss = parseq_permutation_loss(model, images, texts)
            loss = batch_loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()

        should_step = (
            batch_index % args.gradient_accumulation_steps == 0
            or batch_index == len(dataloader)
        )
        if should_step:
            optimizer_step(model, optimizer, scheduler, scaler, args)

        batch_size = images.size(0)
        loss_value = float(batch_loss.detach().item())
        total_loss += loss_value * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return total_loss / total_samples


def evaluate(
    model: Any,
    dataloader: DataLoader,
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
        for batch in tqdm(dataloader, desc="Validacao", leave=False):
            images = batch["images"].to(device)
            texts = batch["texts"]
            _, loss, _ = model.forward_logits_loss(images, texts)
            logits = model(images, max_length=args.max_generation_length)
            batch_predictions, _ = decode_parseq_logits(model, logits)

            batch_size = images.size(0)
            start_index = total_samples
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            references.extend(texts)
            predictions.extend(batch_predictions)
            prediction_rows.extend(
                {
                    "row_index": start_index + offset,
                    "text_true": reference,
                    "text_pred": prediction,
                }
                for offset, (reference, prediction) in enumerate(
                    zip(texts, batch_predictions)
                )
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


def validate_args(args: argparse.Namespace, model: Any) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs precisa ser maior que zero.")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch e gradient accumulation precisam ser maiores que zero.")
    if args.num_beams != 1:
        raise ValueError("PARSeq suporta apenas --num-beams 1 neste pipeline.")
    expected_size = parseq_input_size(model)
    requested_size = (args.image_height, args.image_width)
    if requested_size != expected_size:
        raise ValueError(
            f"A variante {args.model_name} requer imagens {expected_size[0]}x"
            f"{expected_size[1]}, mas foi solicitado {requested_size[0]}x"
            f"{requested_size[1]}."
        )
    model_limit = parseq_max_label_length(model)
    if args.max_target_length > model_limit or args.max_generation_length > model_limit:
        raise ValueError("Limites solicitados excedem a adaptacao posicional do modelo.")


def main() -> None:
    args = parse_args()
    train_csv, val_csv = resolve_split_paths(
        dataset=args.dataset,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
    )
    if args.run_name is None:
        args.run_name = default_run_name(args.dataset, args.model_name)
    args.run_name = sanitize_run_name(args.run_name)

    seed_everything(args.seed)
    device = choose_device()
    output_dir = resolve_output_dir(args.run_name, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_official_parseq(
        model_name=args.model_name,
        hub_repo=args.hub_repo,
        pretrained=args.pretrained,
    )
    adapt_parseq_charset(model, charset_with_extras(model, args.extra_characters))
    adapt_parseq_dimensions(
        model,
        (args.image_height, args.image_width),
        max(args.max_target_length, args.max_generation_length),
    )
    validate_args(args, model)
    model.rng = np.random.default_rng(args.seed)
    configure_parseq_gradient_checkpointing(model, args.gradient_checkpointing)
    freeze_parseq_encoder(model, args.freeze_encoder)
    model.to(device)

    train_transform = (
        None
        if args.no_augmentation
        else build_train_transform(args.augmentation_strength)
    )
    train_dataset = PARSeqSplitDataset(
        csv_path=train_csv,
        data_root=args.data_root,
        label_column=args.label_column,
        image_column=args.image_column,
        image_size=(args.image_height, args.image_width),
        split="train",
        image_transform=train_transform,
        limit=args.limit_train_samples,
    )
    val_dataset = PARSeqSplitDataset(
        csv_path=val_csv,
        data_root=args.data_root,
        label_column=args.label_column,
        image_column=args.image_column,
        image_size=(args.image_height, args.image_width),
        split="val",
        limit=args.limit_val_samples,
    )
    validate_parseq_labels(
        train_dataset.rows[args.label_column].tolist(),
        model,
        args.max_target_length,
        "train",
    )
    validate_parseq_labels(
        val_dataset.rows[args.label_column].tolist(),
        model,
        args.max_target_length,
        "val",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=parseq_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=parseq_collate,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler, total_steps, warmup_steps = build_scheduler(
        optimizer, train_dataloader, args
    )
    scaler = build_grad_scaler(enabled=bool(args.fp16 and device.type == "cuda"))

    history_csv_path = output_dir / "training_history.csv"
    run_config_csv_path = output_dir / "run_config.csv"
    run_summary_csv_path = output_dir / "run_summary.csv"
    text_distribution_csv_path = output_dir / "text_distribution.csv"
    character_distribution_csv_path = output_dir / "character_distribution.csv"
    best_predictions_csv_path = output_dir / "best_val_predictions.csv"
    final_model_dir = output_dir / "parseq_final"
    best_model_dir = output_dir / "parseq_best"

    save_text_distribution_csv(train_dataset, val_dataset, text_distribution_csv_path)
    save_character_distribution_csv(
        train_dataset, val_dataset, character_distribution_csv_path
    )
    save_run_config_csv(
        args,
        output_dir,
        device,
        train_csv,
        val_csv,
        train_dataset,
        val_dataset,
        model,
        run_config_csv_path,
    )

    print(f"Nome da execucao: {args.run_name}")
    print(f"Modelo oficial: {args.hub_repo}/{args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Pasta da execucao: {output_dir}")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Amostras de treino: {len(train_dataset)}")
    print(f"Amostras de validacao: {len(val_dataset)}")
    print(f"Passos de otimizacao: {total_steps} (warmup: {warmup_steps})")
    print(f"Data augmentation: {'desligado' if args.no_augmentation else 'ligado'}")

    best_val_cer = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_val_loss: float | None = None
    history_rows: list[dict[str, Any]] = []

    def metadata(**extra: Any) -> dict[str, Any]:
        base = {
            "run_name": args.run_name,
            "model_kind": "parseq",
            "model_name": args.model_name,
            "hub_repo": args.hub_repo,
            "pretrained": args.pretrained,
            "dataset": args.dataset,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "label_column": args.label_column,
            "image_column": args.image_column,
            "image_size": [args.image_height, args.image_width],
            "max_target_length": args.max_target_length,
            "max_generation_length": args.max_generation_length,
            "num_beams": args.num_beams,
            "charset": tokenizer_charset(model.tokenizer),
            "gradient_checkpointing": args.gradient_checkpointing,
        }
        base.update(extra)
        return base

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, scaler, device, args
        )
        val_loss, val_metrics, val_predictions = evaluate(
            model, val_dataloader, device, args
        )
        val_accuracy = float(val_metrics["exact_match_accuracy"])
        val_cer = float(val_metrics["cer"])
        val_wer = float(val_metrics["wer"])
        print(
            f"Epoch {epoch:03d}/{args.epochs} - loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f} "
            f"- val_cer: {val_cer:.4f} - val_wer: {val_wer:.4f}"
        )

        is_best_epoch = (val_cer < best_val_cer) or (
            val_cer == best_val_cer and val_accuracy > best_val_accuracy
        )
        if is_best_epoch:
            best_val_cer = val_cer
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            save_parseq_bundle(
                model,
                best_model_dir,
                metadata(
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    best_val_accuracy=best_val_accuracy,
                    best_val_cer=best_val_cer,
                    best_val_wer=val_wer,
                ),
            )
            val_predictions.to_csv(best_predictions_csv_path, index=False)
            print(
                f"Novo melhor PARSeq salvo: epoch {best_epoch} "
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

    save_parseq_bundle(
        model,
        final_model_dir,
        metadata(
            epochs_completed=len(history_rows),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            best_val_cer=best_val_cer,
        ),
    )

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
        f"(epoch {best_epoch}, val_cer {best_val_cer:.4f})"
    )


if __name__ == "__main__":
    main()
