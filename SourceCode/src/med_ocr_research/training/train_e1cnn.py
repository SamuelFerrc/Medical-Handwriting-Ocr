from __future__ import annotations

import argparse
import json
import random
import re
import sys
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

from med_ocr_research.models.E1Cnn import E1Cnn
from med_ocr_research.models.E2CnnBiLSTM import E2CnnBiLSTM


DEFAULT_TRAIN_CSV = PROJECT_ROOT / "data/splits/scenarios/prescriptionbd/train.csv"
DEFAULT_VAL_CSV = PROJECT_ROOT / "data/splits/scenarios/prescriptionbd/val.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs/runs"
DEFAULT_RUN_NAME = "e1cnn"
DEFAULT_MODEL_KIND = "e1cnn"

MODEL_REGISTRY = {
    "e1cnn": {
        "display_name": "E1Cnn",
        "artifact_prefix": "e1cnn",
        "default_run_name": DEFAULT_RUN_NAME,
        "class": E1Cnn,
    },
    "e2cnn-bilstm": {
        "display_name": "E2CnnBiLSTM",
        "artifact_prefix": "e2cnn_bilstm",
        "default_run_name": "e2cnn_bilstm",
        "class": E2CnnBiLSTM,
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="medium",
    )
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
    if transforms is None:
        raise RuntimeError(
            "torchvision nao esta instalado. Instale requirements-train.txt "
            "ou rode com --no-augmentation."
        )

    configs = {
        "light": {
            "degrees": 4,
            "translate": (0.02, 0.03),
            "scale": (0.97, 1.03),
            "shear": 2,
            "brightness": 0.10,
            "contrast": 0.15,
            "affine_p": 0.7,
            "jitter_p": 0.4,
            "blur_p": 0.1,
        },
        "medium": {
            "degrees": 8,
            "translate": (0.05, 0.06),
            "scale": (0.90, 1.08),
            "shear": 5,
            "brightness": 0.20,
            "contrast": 0.25,
            "affine_p": 0.8,
            "jitter_p": 0.5,
            "blur_p": 0.25,
        },
        "strong": {
            "degrees": 10,
            "translate": (0.07, 0.08),
            "scale": (0.85, 1.12),
            "shear": 7,
            "brightness": 0.25,
            "contrast": 0.35,
            "affine_p": 0.9,
            "jitter_p": 0.6,
            "blur_p": 0.3,
        },
    }
    config = configs[strength]

    return transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=config["degrees"],
                        translate=config["translate"],
                        scale=config["scale"],
                        shear=config["shear"],
                        fill=255,
                    )
                ],
                p=config["affine_p"],
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=config["brightness"],
                        contrast=config["contrast"],
                    )
                ],
                p=config["jitter_p"],
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=config["blur_p"],
            ),
        ]
    )


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
        image_transform: transforms.Compose | None = None,
    ) -> None:
        self.data_root = data_root
        self.label_column = label_column
        self.image_column = image_column
        self.image_size = image_size
        self.image_transform = image_transform

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
        if class_to_idx is None:
            labels = sorted(rows["_target_label"].unique())
            self.class_to_idx = {label: index for index, label in enumerate(labels)}
        else:
            self.class_to_idx = class_to_idx
            rows = rows[rows["_target_label"].isin(class_to_idx)].copy()
            if rows.empty:
                raise ValueError(
                    f"Nenhuma amostra de {csv_path} possui palavra presente no treino"
                )

        self.idx_to_class = {index: label for label, index in self.class_to_idx.items()}
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        label_tensor = torch.tensor(
            self.class_to_idx[str(row["_target_label"])],
            dtype=torch.long,
        )
        return image_tensor, label_tensor


def infer_num_outputs(model: nn.Module, device: torch.device, image_size: tuple[int, int]) -> int:
    model.eval()
    with torch.no_grad():
        sample = torch.zeros(1, 1, image_size[0], image_size[1], device=device)
        return int(model(sample).shape[1])


def build_model(model_kind: str, num_classes: int) -> nn.Module:
    model_config = get_model_config(model_kind)
    model_class = model_config["class"]
    model = model_class(num_classes=num_classes)
    return model


def save_class_distribution_csv(
    train_dataset: TrainingSplitDataset,
    val_dataset: TrainingSplitDataset,
    output_path: Path,
) -> None:
    rows = []
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        counts = dataset.rows["_target_label"].value_counts().to_dict()
        total = len(dataset)
        for label, class_index in dataset.class_to_idx.items():
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
                "output_dir": str(output_dir),
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "data_root": str(args.data_root),
                "label_column": args.label_column,
                "image_column": args.image_column,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
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
                "num_classes": len(train_dataset.class_to_idx),
                "model_outputs": num_outputs,
            }
        ]
    ).to_csv(output_path, index=False)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def main(default_model_kind: str = DEFAULT_MODEL_KIND) -> None:
    args = parse_args(default_model_kind=default_model_kind)
    model_config = get_model_config(args.model_kind)
    if args.run_name is None:
        args.run_name = str(model_config["default_run_name"])
    model_name = str(model_config["display_name"])
    artifact_prefix = str(model_config["artifact_prefix"])
    seed_everything(args.seed)
    output_dir = resolve_output_dir(args.run_name, args.output_dir)

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
        image_transform=train_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_dataset = TrainingSplitDataset(
        csv_path=args.val_csv,
        data_root=args.data_root,
        label_column=args.label_column,
        image_column=args.image_column,
        image_size=(args.image_height, args.image_width),
        split="val",
        class_to_idx=dataset.class_to_idx,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_kind=args.model_kind,
        num_classes=len(dataset.class_to_idx),
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

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
    print(f"Palavras/classes: {len(dataset.class_to_idx)}")
    print(f"Saidas da {model_name}: {num_outputs}")
    print(f"CSV de treino: {args.train_csv}")
    print(f"CSV de validacao: {args.val_csv}")
    print(f"Data augmentation: {'desligado' if args.no_augmentation else 'ligado'}")
    if not args.no_augmentation:
        print(f"Forca da augmentation: {args.augmentation_strength}")

    best_val_accuracy = -1.0
    best_val_loss = None
    best_epoch = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"- loss: {train_loss:.4f} "
            f"- accuracy: {train_accuracy:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_accuracy: {val_accuracy:.4f}"
        )

        is_best_epoch = val_accuracy > best_val_accuracy
        if is_best_epoch:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "run_name": args.run_name,
                    "model_kind": args.model_kind,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": dataset.class_to_idx,
                    "idx_to_class": dataset.idx_to_class,
                    "image_size": (args.image_height, args.image_width),
                    "train_csv": str(args.train_csv),
                    "val_csv": str(args.val_csv),
                    "label_column": args.label_column,
                    "image_column": args.image_column,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(
                f"Novo melhor checkpoint salvo: epoch {best_epoch} "
                f"- val_accuracy: {best_val_accuracy:.4f}"
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "is_best_epoch": is_best_epoch,
                "best_epoch_so_far": best_epoch,
                "best_val_accuracy_so_far": best_val_accuracy,
                "best_val_loss_so_far": best_val_loss,
            }
        )
        pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)

    torch.save(
        {
            "run_name": args.run_name,
            "model_kind": args.model_kind,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
            "idx_to_class": dataset.idx_to_class,
            "image_size": (args.image_height, args.image_width),
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "label_column": args.label_column,
            "image_column": args.image_column,
        },
        checkpoint_path,
    )
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
                    "best_val_loss": best_val_loss,
                    "final_train_loss": final_metrics["train_loss"],
                    "final_train_accuracy": final_metrics["train_accuracy"],
                    "final_val_loss": final_metrics["val_loss"],
                    "final_val_accuracy": final_metrics["val_accuracy"],
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
        f"(epoch {best_epoch}, val_accuracy: {best_val_accuracy:.4f})"
    )
    print(f"Mapa de classes salvo em: {class_map_path}")
    print(f"Historico de treino salvo em: {history_csv_path}")
    print(f"Resumo da execucao salvo em: {run_summary_csv_path}")
    print(f"Distribuicao de classes salva em: {class_distribution_csv_path}")
    print(f"Configuracao da execucao salva em: {run_config_csv_path}")


if __name__ == "__main__":
    main()
