from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "src/med_ocr_research/training/train_e2cnn_bilstm.py"
SCENARIO_RUNS = (
    ("e2.01", "prescriptionbd"),
    ("e2.02", "rxhandbd"),
    ("e2.03", "combined"),
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Treina E2CnnBiLSTM em sequencia: e2.01 prescriptionbd, "
            "e2.02 rxhandbd e e2.03 combined."
        )
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="strong",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpretador Python usado para chamar o script de treino.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra os comandos sem iniciar o treino.",
    )
    return parser.parse_known_args()


def build_command(args: argparse.Namespace, run_name: str, scenario: str) -> list[str]:
    split_dir = PROJECT_ROOT / "data/splits/scenarios" / scenario
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    missing = [path for path in (train_csv, val_csv) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Split ausente para {scenario}: {missing_paths}")

    command = [
        args.python,
        str(TRAIN_SCRIPT),
        "--run-name",
        run_name,
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--data-root",
        str(args.data_root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--image-height",
        str(args.image_height),
        "--image-width",
        str(args.image_width),
        "--label-column",
        args.label_column,
        "--image-column",
        args.image_column,
    ]
    if args.no_augmentation:
        command.append("--no-augmentation")
    else:
        command.extend(["--augmentation-strength", args.augmentation_strength])
    return command


def main() -> None:
    args, extra_args = parse_args()
    for run_name, scenario in SCENARIO_RUNS:
        command = build_command(args, run_name=run_name, scenario=scenario)
        command.extend(extra_args)
        print()
        print(f"=== Treinando {run_name}: {scenario} ===")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
