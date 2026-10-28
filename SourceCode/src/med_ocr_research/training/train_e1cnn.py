from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms
except ImportError:
    transforms = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.experiments.seeds import ensure_new_run, seeded_run_name
from med_ocr_research.evaluation.metrics import aggregate_metrics
from med_ocr_research.models.E1Cnn import E1Cnn
from med_ocr_research.models.E2CnnBiLSTM import E2CnnBiLSTM
from med_ocr_research.training.augmentation import build_handwriting_train_transform


DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data/splits/scenarios/prescriptionbd/train.csv"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/splits/scenarios/prescriptionbd/val.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs/runs"
DEFAULT_RUN_NAME = "e1cnn"
DEFAULT_MODEL_KIND = "e1cnn"
CTC_BLANK_TOKEN = "<blank>"
CLASSIFICATION_TARGET = "class"
CTC_TARGET = "ctc"

MODEL_REGISTRY = {
    "e1cnn": {
        "display_name": "E1Cnn",
        "artifact_prefix": "e1cnn",
        "default_run_name": DEFAULT_RUN_NAME,
        "class": E1Cnn,
        "target_kind": CLASSIFICATION_TARGET,
    },
    "e2cnn-bilstm": {
        "display_name": "E2CnnBiLSTM",
        "artifact_prefix": "e2cnn_bilstm",
        "default_run_name": "e2cnn_bilstm",
        "class": E2CnnBiLSTM,
        "target_kind": CTC_TARGET,
    },
}


def get_model_config(model_kind: str) -> dict[str, object]:
    try:
        return MODEL_REGISTRY[model_kind]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Modelo nao suportado: {model_kind}. Opcoes: {supported}"
        ) from exc


def get_target_kind(model_kind: str) -> str:
    return str(get_model_config(model_kind)["target_kind"])


def is_ctc_model(model_kind: str) -> bool:
    return get_target_kind(model_kind) == CTC_TARGET


def parse_args(default_model_kind: str = DEFAULT_MODEL_KIND) -> argparse.Namespace:
    default_model_config = get_model_config(default_model_kind)
    parser = argparse.ArgumentParser(
        description=(
            "Treina classificadores CNN usando os splits de treino e validacao "
            f"do dataset. Padrao: {default_model_config['display_name']}."
        )
    )
    parser.add_argument(
        "--model-kind",
        choices=tuple(MODEL_REGISTRY),
        default=default_model_kind,
        help="Arquitetura que sera treinada.",
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Padrao: e1cnn para E1Cnn, e2cnn_bilstm para E2CnnBiLSTM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sobrescreve outputs/runs/<run-name> quando informado.",
    )
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="strong",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol-version", default="custom")
    parser.add_argument("--protocol-fingerprint", default="")
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


def resolve_output_dir(run_name: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return DEFAULT_RUNS_DIR / sanitize_run_name(run_name)


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    return data_root / path


def build_train_transform(strength: str) -> transforms.Compose:
    return build_handwriting_train_transform(transforms, strength)


class TrainingSplitDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        label_column: str,
        image_column: str,
        image_size: tuple[int, int],
        split: str | None = None,
        class_to_idx: dict[str, int] | None = None,
        char_to_idx: dict[str, int] | None = None,
        target_kind: str = CLASSIFICATION_TARGET,
        image_transform: transforms.Compose | None = None,
        return_text: bool = False,
    ) -> None:
        self.data_root = data_root
        self.label_column = label_column
        self.image_column = image_column
        self.image_size = image_size
        self.image_transform = image_transform
        self.target_kind = target_kind
        self.return_text = return_text

        rows = pd.read_csv(csv_path)
        if split is not None and "split" in rows.columns:
            rows = rows[rows["split"].astype(str).str.lower() == split].copy()

        required = {image_column, label_column}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")
        if rows.empty:
            raise ValueError(f"Nenhuma amostra de treino encontrada em {csv_path}")

        rows[label_column] = rows[label_column].astype(str)
        rows["_target_label"] = rows[label_column]
        if target_kind == CLASSIFICATION_TARGET:
            if class_to_idx is None:
                labels = sorted(rows["_target_label"].unique())
                self.class_to_idx = {label: index for index, label in enumerate(labels)}
            else:
                self.class_to_idx = class_to_idx
            rows["_loss_representable"] = rows["_target_label"].isin(self.class_to_idx)
            self.idx_to_class = {
                index: label for label, index in self.class_to_idx.items()
            }
            self.char_to_idx: dict[str, int] = {}
            self.idx_to_char: dict[int, str] = {}
            self.blank_idx: int | None = None
        elif target_kind == CTC_TARGET:
            if char_to_idx is None:
                characters = sorted(
                    {char for label in rows["_target_label"] for char in label}
                )
                self.char_to_idx = {CTC_BLANK_TOKEN: 0}
                self.char_to_idx.update(
                    {char: index for index, char in enumerate(characters, start=1)}
                )
            else:
                if CTC_BLANK_TOKEN not in char_to_idx:
                    raise ValueError(
                        f"O vocabulario CTC precisa conter {CTC_BLANK_TOKEN!r}."
                    )
                self.char_to_idx = char_to_idx
            valid_characters = set(self.char_to_idx).difference({CTC_BLANK_TOKEN})
            rows["_loss_representable"] = rows["_target_label"].map(
                lambda label: all(char in valid_characters for char in label)
            )
            self.class_to_idx = self.char_to_idx
            self.idx_to_char = {
                index: char for char, index in self.char_to_idx.items()
            }
            self.idx_to_class = self.idx_to_char
            self.blank_idx = self.char_to_idx[CTC_BLANK_TOKEN]
        else:
            raise ValueError(f"target_kind nao suportado: {target_kind}")

        self.rows = rows.reset_index(drop=True)
        self.representable_samples = int(self.rows["_loss_representable"].sum())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image_path = resolve_image_path(self.data_root, row[self.image_column])

        with Image.open(image_path) as image:
            if image.mode == "RGB":
                image = image.convert("L")
            elif image.mode != "L":
                image = image.convert("L")
            image = image.resize((self.image_size[1], self.image_size[0]))
            if self.image_transform is not None:
                image = self.image_transform(image)
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        label = str(row["_target_label"])
        if self.target_kind == CTC_TARGET:
            label_tensor = torch.tensor(
                [self.char_to_idx.get(char, -1) for char in label],
                dtype=torch.long,
            )
            return image_tensor, label_tensor, label

        label_tensor = torch.tensor(
            self.class_to_idx.get(label, -1),
            dtype=torch.long,
        )
        if self.return_text:
            return image_tensor, label_tensor, label
        return image_tensor, label_tensor


def ctc_collate(batch):
    images, targets, texts = zip(*batch)
    target_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
    if targets:
        concatenated_targets = torch.cat(targets)
    else:
        concatenated_targets = torch.empty(0, dtype=torch.long)
    return (
        torch.stack(images),
        concatenated_targets,
        target_lengths,
        list(texts),
    )


def infer_num_outputs(model: nn.Module, device: torch.device, image_size: tuple[int, int]) -> int:
    model.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 1, image_size[0], image_size[1], device=device)
        logits = model(sample)
        return int(logits.shape[-1])


def build_model(model_kind: str, num_classes: int, blank_idx: int = 0) -> nn.Module:
    model_config = get_model_config(model_kind)
    model_class = model_config["class"]
    if get_target_kind(model_kind) == CTC_TARGET:
        return model_class(
            num_classes=num_classes,
            blank_idx=blank_idx,
            default_return_sequence=True,
        )
    model = model_class(num_classes=num_classes)
    return model


def save_class_distribution_csv(
    train_dataset: TrainingSplitDataset,
    val_dataset: TrainingSplitDataset,
    output_path: Path,
) -> None:
    rows = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if dataset.target_kind == CTC_TARGET:
            counts: dict[str, int] = {}
            for label in dataset.rows["_target_label"]:
                for char in str(label):
                    counts[char] = counts.get(char, 0) + 1
            total = sum(counts.values())
        else:
            counts = dataset.rows["_target_label"].value_counts().to_dict()
            total = len(dataset)

        labels = dict(dataset.class_to_idx)
        labels.update({label: -1 for label in counts if label not in labels})
        for label, class_index in labels.items():
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "split": split_name,
                    "class_index": class_index,
                    "label": label,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_run_config_csv(
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    num_outputs: int,
    train_dataset: TrainingSplitDataset,
    val_dataset: TrainingSplitDataset,
    output_path: Path,
) -> None:
    pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "model_kind": args.model_kind,
                "model_name": get_model_config(args.model_kind)["display_name"],
                "artifact_prefix": get_model_config(args.model_kind)[
                    "artifact_prefix"
                ],
                "target_kind": get_target_kind(args.model_kind),
                "blank_idx": train_dataset.blank_idx,
                "output_dir": str(output_dir),
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "data_root": str(args.data_root),
                "label_column": args.label_column,
                "image_column": args.image_column,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_ratio": args.warmup_ratio,
                "max_grad_norm": args.max_grad_norm,
                "fp16": args.fp16,
                "optimizer": "AdamW",
                "scheduler": "linear_warmup",
                "checkpoint_selection": "min_val_cer_then_max_val_accuracy",
                "protocol_version": args.protocol_version,
                "protocol_fingerprint": args.protocol_fingerprint,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_workers": args.num_workers,
                "image_height": args.image_height,
                "image_width": args.image_width,
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
                "val_vocabulary_representable_samples": val_dataset.representable_samples,
                "validation_population": "complete_split",
                "validation_loss_population": "representable_targets_only",
                "num_classes": len(train_dataset.class_to_idx),
                "model_outputs": num_outputs,
            }
        ]
    ).to_csv(output_path, index=False)


def build_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_scheduler(optimizer, dataloader: DataLoader, args: argparse.Namespace):
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), total_steps, warmup_steps


def select_ctc_loss_targets(logits, targets, target_lengths):
    """Select only representable CTC targets; retain all logits for decoding."""
    segments = torch.split(targets, target_lengths.detach().cpu().tolist())
    eligible = []
    for index, segment in enumerate(segments):
        repeats = int((segment[1:] == segment[:-1]).sum().item())
        if bool((segment >= 0).all()) and segment.numel() + repeats <= logits.size(1):
            eligible.append(index)
    if not eligible:
        return None
    indices = torch.tensor(eligible, dtype=torch.long, device=logits.device)
    return (
        logits.index_select(0, indices).float(),
        torch.cat([segments[index] for index in eligible]),
        target_lengths.index_select(0, indices),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_kind: str = CLASSIFICATION_TARGET,
    idx_to_token: dict[int, str] | None = None,
    gradient_accumulation_steps: int = 1,
    scheduler=None,
    scaler=None,
    max_grad_norm: float = 1.0,
    fp16: bool = False,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    window_samples = 0
    scaler = scaler if scaler is not None else build_grad_scaler(enabled=False)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(dataloader, start=1):
        images = batch[0].to(device)
        batch_size = images.size(0)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if fp16 and device.type == "cuda" else nullcontext()
        )
        with autocast_context:
            logits = model(images)
            if target_kind == CTC_TARGET:
                _, targets, target_lengths, texts = batch
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                selected = select_ctc_loss_targets(logits, targets, target_lengths)
                if selected is None or selected[0].size(0) != batch_size:
                    raise ValueError("Treino CTC contem alvos fora do vocabulario ou longos demais.")
                loss = model.ctc_loss(
                    logits=selected[0], targets=selected[1], target_lengths=selected[2],
                )
            else:
                labels = batch[1].to(device)
                if criterion is None:
                    raise ValueError("criterion precisa ser informado para classificacao.")
                if bool((labels < 0).any()):
                    raise ValueError("Treino contem palavras fora do vocabulario.")
                loss = criterion(logits, labels)

        # Accumulate sums, then normalize by the actual window size. This handles
        # both an uneven last microbatch and an incomplete accumulation window.
        scaler.scale(loss * batch_size).backward()
        window_samples += batch_size
        if batch_index % gradient_accumulation_steps == 0 or batch_index == len(dataloader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and (
                not scaler.is_enabled() or scaler.get_scale() >= scale_before_step
            ):
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0

        total_loss += loss.detach().item() * batch_size
        if target_kind == CTC_TARGET:
            predictions = model.ctc_decode(logits.detach(), idx_to_token=idx_to_token)
            total_correct += sum(str(prediction) == str(target) for prediction, target in zip(predictions, texts))
        else:
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    target_kind: str = CLASSIFICATION_TARGET,
    idx_to_token: dict[int, str] | None = None,
) -> dict[str, float | int]:
    """Score the complete validation split, with loss on representable targets."""
    model.eval()
    total_loss = 0.0
    loss_samples = 0
    references: list[str] = []
    hypotheses: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            logits = model(images)
            if target_kind == CTC_TARGET:
                _, targets, target_lengths, texts = batch
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                selected = select_ctc_loss_targets(logits, targets, target_lengths)
                if selected is not None:
                    loss = model.ctc_loss(
                        logits=selected[0], targets=selected[1], target_lengths=selected[2],
                    )
                    selected_samples = selected[0].size(0)
                    total_loss += loss.item() * selected_samples
                    loss_samples += selected_samples
                predictions = model.ctc_decode(logits, idx_to_token=idx_to_token)
            else:
                if len(batch) != 3 or idx_to_token is None:
                    raise ValueError("Validacao requer textos originais e mapa de classes.")
                _, labels, texts = batch
                labels = labels.to(device)
                if criterion is None:
                    raise ValueError("criterion precisa ser informado para classificacao.")
                eligible = labels >= 0
                selected_samples = int(eligible.sum().item())
                if selected_samples:
                    loss = criterion(logits[eligible], labels[eligible])
                    total_loss += loss.item() * selected_samples
                    loss_samples += selected_samples
                predictions = [idx_to_token[int(index)] for index in logits.argmax(dim=1).tolist()]
            references.extend(str(text) for text in texts)
            hypotheses.extend(str(prediction) for prediction in predictions)

    metrics = aggregate_metrics(references, hypotheses)
    return {
        "loss": total_loss / loss_samples if loss_samples else float("nan"),
        "loss_samples": loss_samples,
        "samples": metrics.samples,
        "accuracy": metrics.exact_match_accuracy,
        "cer": metrics.cer,
        "wer": metrics.wer,
    }


def main(default_model_kind: str = DEFAULT_MODEL_KIND) -> None:
    args = parse_args(default_model_kind=default_model_kind)
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("epochs, batch-size e gradient-accumulation-steps devem ser positivos.")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("warmup-ratio precisa estar entre 0 e 1.")
    model_config = get_model_config(args.model_kind)
    target_kind = get_target_kind(args.model_kind)
    collate_fn = ctc_collate if target_kind == CTC_TARGET else None
    if args.run_name is None:
        args.run_name = str(model_config["default_run_name"])
    args.run_name = seeded_run_name(sanitize_run_name(args.run_name), args.seed)
    model_name = str(model_config["display_name"])
    artifact_prefix = str(model_config["artifact_prefix"])
    seed_everything(args.seed)
    output_dir = resolve_output_dir(args.run_name, args.output_dir)
    ensure_new_run(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA disponivel: usando GPU {torch.cuda.get_device_name(0)}")

    train_transform = (
        None
        if args.no_augmentation
        else build_train_transform(args.augmentation_strength)
    )
    dataset = TrainingSplitDataset(
        csv_path=args.train_csv,
        data_root=args.data_root,
        label_column=args.label_column,
        image_column=args.image_column,
        image_size=(args.image_height, args.image_width),
        split="train",
        target_kind=target_kind,
        image_transform=train_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_dataset = TrainingSplitDataset(
        csv_path=args.val_csv,
        data_root=args.data_root,
        label_column=args.label_column,
        image_column=args.image_column,
        image_size=(args.image_height, args.image_width),
        split="val",
        class_to_idx=dataset.class_to_idx,
        char_to_idx=dataset.char_to_idx if target_kind == CTC_TARGET else None,
        target_kind=target_kind,
        return_text=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = build_model(
        model_kind=args.model_kind,
        num_classes=len(dataset.class_to_idx),
        blank_idx=dataset.blank_idx or 0,
    ).to(device)
    num_outputs = infer_num_outputs(
        model,
        device=device,
        image_size=(args.image_height, args.image_width),
    )
    if len(dataset.class_to_idx) != num_outputs:
        raise ValueError(
            f"A {model_name} retorna "
            f"{num_outputs} classes, mas o alvo possui {len(dataset.class_to_idx)}. "
            "Ajuste num_classes na criacao do modelo."
        )

    criterion = None if target_kind == CTC_TARGET else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler, total_steps, warmup_steps = build_scheduler(optimizer, dataloader, args)
    scaler = build_grad_scaler(enabled=bool(args.fp16 and device.type == "cuda"))
    training_metadata = {
        "seed": args.seed,
        "protocol_version": args.protocol_version,
        "protocol_fingerprint": args.protocol_fingerprint,
        "checkpoint_selection": "min_val_cer_then_max_val_accuracy",
        "validation_population": "complete_split",
        "validation_loss_population": "representable_targets_only",
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": "AdamW",
        "scheduler": "linear_warmup",
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "fp16": args.fp16,
        "optimizer_steps_planned": total_steps,
        "warmup_steps": warmup_steps,
        "train_samples": len(dataset),
        "val_samples": len(val_dataset),
        "val_vocabulary_representable_samples": val_dataset.representable_samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    class_map_path = output_dir / f"{artifact_prefix}_class_to_idx.json"
    checkpoint_path = output_dir / f"{artifact_prefix}_training_checkpoint.pt"
    best_checkpoint_path = output_dir / f"{artifact_prefix}_best_checkpoint.pt"
    history_csv_path = output_dir / "training_history.csv"
    run_config_csv_path = output_dir / "run_config.csv"
    run_summary_csv_path = output_dir / "run_summary.csv"
    class_distribution_csv_path = output_dir / "class_distribution.csv"
    class_map_path.write_text(
        json.dumps(dataset.class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_class_distribution_csv(
        train_dataset=dataset,
        val_dataset=val_dataset,
        output_path=class_distribution_csv_path,
    )
    save_run_config_csv(
        args=args,
        output_dir=output_dir,
        device=device,
        num_outputs=num_outputs,
        train_dataset=dataset,
        val_dataset=val_dataset,
        output_path=run_config_csv_path,
    )

    print(f"Nome da execucao: {args.run_name}")
    print(f"Modelo: {model_name} ({args.model_kind})")
    print(f"Pasta da execucao: {output_dir}")
    print(f"Dispositivo: {device}")
    print(f"Amostras de treino: {len(dataset)}")
    print(f"Amostras de validacao: {len(val_dataset)}")
    print(f"Alvos de validacao no vocabulario: {val_dataset.representable_samples}")
    print(f"Batch efetivo: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Passos de otimizacao: {total_steps} (warmup: {warmup_steps})")
    if target_kind == CTC_TARGET:
        print(f"Caracteres/classes CTC: {len(dataset.class_to_idx)}")
        print(f"Blank CTC: {dataset.blank_idx} ({CTC_BLANK_TOKEN})")
    else:
        print(f"Palavras/classes: {len(dataset.class_to_idx)}")
    print(f"Saidas da {model_name}: {num_outputs}")
    print(f"CSV de treino: {args.train_csv}")
    print(f"CSV de validacao: {args.val_csv}")
    print(f"Data augmentation: {'desligado' if args.no_augmentation else 'ligado'}")
    if not args.no_augmentation:
        print(f"Forca da augmentation: {args.augmentation_strength}")

    best_val_accuracy = -1.0
    best_val_cer = float("inf")
    best_val_wer = None
    best_val_loss = None
    best_val_loss_samples = 0
    best_epoch = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            target_kind=target_kind,
            idx_to_token=dataset.idx_to_char if target_kind == CTC_TARGET else None,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scheduler=scheduler,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            fp16=args.fp16,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            target_kind=target_kind,
            idx_to_token=dataset.idx_to_class,
        )
        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        val_cer = float(val_metrics["cer"])
        val_wer = float(val_metrics["wer"])
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"- loss: {train_loss:.4f} "
            f"- accuracy: {train_accuracy:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_accuracy: {val_accuracy:.4f} "
            f"- val_cer: {val_cer:.4f} - val_wer: {val_wer:.4f} "
            f"- val_loss_samples: {val_metrics['loss_samples']}/{val_metrics['samples']}"
        )

        is_best_epoch = (val_cer < best_val_cer) or (
            val_cer == best_val_cer and val_accuracy > best_val_accuracy
        )
        if is_best_epoch:
            best_val_accuracy = val_accuracy
            best_val_cer = val_cer
            best_val_wer = val_wer
            best_val_loss = val_loss
            best_val_loss_samples = int(val_metrics["loss_samples"])
            best_epoch = epoch
            torch.save(
                {
                    **training_metadata,
                    "run_name": args.run_name,
                    "model_kind": args.model_kind,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": dataset.class_to_idx,
                    "idx_to_class": dataset.idx_to_class,
                    "target_kind": target_kind,
                    "blank_token": CTC_BLANK_TOKEN if target_kind == CTC_TARGET else None,
                    "blank_idx": dataset.blank_idx,
                    "char_to_idx": dataset.char_to_idx,
                    "idx_to_char": dataset.idx_to_char,
                    "image_size": (args.image_height, args.image_width),
                    "train_csv": str(args.train_csv),
                    "val_csv": str(args.val_csv),
                    "label_column": args.label_column,
                    "image_column": args.image_column,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": val_loss,
                    "best_val_cer": best_val_cer,
                    "best_val_wer": best_val_wer,
                    "val_loss_samples": best_val_loss_samples,
                },
                best_checkpoint_path,
            )
            print(
                f"Novo melhor checkpoint salvo: epoch {best_epoch} "
                f"- val_cer: {best_val_cer:.4f}"
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_cer": val_cer,
                "val_wer": val_wer,
                "val_samples": val_metrics["samples"],
                "val_loss_samples": val_metrics["loss_samples"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best_epoch": is_best_epoch,
                "best_epoch_so_far": best_epoch,
                "best_val_accuracy_so_far": best_val_accuracy,
                "best_val_loss_so_far": best_val_loss,
                "best_val_cer_so_far": best_val_cer,
                "best_val_wer_so_far": best_val_wer,
            }
        )
        pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)

    torch.save(
        {
            **training_metadata,
            "run_name": args.run_name,
            "model_kind": args.model_kind,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "idx_to_class": dataset.idx_to_class,
            "target_kind": target_kind,
            "blank_token": CTC_BLANK_TOKEN if target_kind == CTC_TARGET else None,
            "blank_idx": dataset.blank_idx,
            "char_to_idx": dataset.char_to_idx,
            "idx_to_char": dataset.idx_to_char,
            "image_size": (args.image_height, args.image_width),
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "label_column": args.label_column,
            "image_column": args.image_column,
            "best_epoch": best_epoch,
            "best_val_cer": best_val_cer,
            "best_val_accuracy": best_val_accuracy,
            "best_val_wer": best_val_wer,
            "best_val_loss": best_val_loss,
            "val_loss_samples": int(val_metrics["loss_samples"]),
        },
        checkpoint_path,
    )
    if history_rows:
        final_metrics = history_rows[-1]
        pd.DataFrame(
            [
                {
                    **training_metadata,
                    "run_name": args.run_name,
                    "output_dir": str(output_dir),
                    "epochs_completed": len(history_rows),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": best_val_loss,
                    "best_val_cer": best_val_cer,
                    "best_val_wer": best_val_wer,
                    "best_val_loss_samples": best_val_loss_samples,
                    "final_train_loss": final_metrics["train_loss"],
                    "final_train_accuracy": final_metrics["train_accuracy"],
                    "final_val_loss": final_metrics["val_loss"],
                    "final_val_accuracy": final_metrics["val_accuracy"],
                    "final_val_cer": final_metrics["val_cer"],
                    "final_val_wer": final_metrics["val_wer"],
                    "final_val_loss_samples": final_metrics["val_loss_samples"],
                    "final_checkpoint_path": str(checkpoint_path),
                    "best_checkpoint_path": str(best_checkpoint_path),
                    "history_csv_path": str(history_csv_path),
                    "class_distribution_csv_path": str(class_distribution_csv_path),
                    "run_config_csv_path": str(run_config_csv_path),
                }
            ]
        ).to_csv(run_summary_csv_path, index=False)

    print(f"Checkpoint salvo em: {checkpoint_path}")
    print(
        f"Melhor checkpoint salvo em: {best_checkpoint_path} "
        f"(epoch {best_epoch}, val_cer: {best_val_cer:.4f})"
    )
    print(f"Mapa de classes salvo em: {class_map_path}")
    print(f"Historico de treino salvo em: {history_csv_path}")
    print(f"Resumo da execucao salvo em: {run_summary_csv_path}")
    print(f"Distribuicao de classes salva em: {class_distribution_csv_path}")
    print(f"Configuracao da execucao salva em: {run_config_csv_path}")


if __name__ == "__main__":
    main()
