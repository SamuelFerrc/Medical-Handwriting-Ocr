from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from med_ocr_research.experiments.seeds import seeded_run_name

TRAIN_SCRIPT = PROJECT_ROOT / "src/med_ocr_research/training/train_trocr.py"
SCENARIO_RUNS = {
    "prescriptionbd": "e3.01",
    "rxhandbd": "e3.02",
    "combined": "e3.03",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Faz fine-tuning do TrOCR em sequencia. Por padrao roda "
            "prescriptionbd, rxhandbd e combined."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SCENARIO_RUNS),
        default=list(SCENARIO_RUNS),
        help="Datasets/cenarios que serao treinados, na ordem informada.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help=(
            "Micro-batch por passo. O padrao e baixo para caber em GPUs de "
            "6 GB; use gradient accumulation para manter batch efetivo maior."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--checkpoint", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--processor-checkpoint", default=None)
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
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
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="strong",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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


def build_command(args: argparse.Namespace, dataset: str) -> list[str]:
    split_dir = PROJECT_ROOT / "data/splits/scenarios" / dataset
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    missing = [path for path in (train_csv, val_csv) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Split ausente para {dataset}: {missing_paths}")

    command = [
        args.python,
        str(TRAIN_SCRIPT),
        "--run-name",
        seeded_run_name(SCENARIO_RUNS[dataset], args.seed),
        "--dataset",
        dataset,
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        args.checkpoint,
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
        "--max-target-length",
        str(args.max_target_length),
        "--max-generation-length",
        str(args.max_generation_length),
        "--num-beams",
        str(args.num_beams),
        "--image-height",
        str(args.image_height),
        "--image-width",
        str(args.image_width),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--label-column",
        args.label_column,
        "--image-column",
        args.image_column,
        "--seed",
        str(args.seed),
    ]
    if args.processor_checkpoint:
        command.extend(["--processor-checkpoint", args.processor_checkpoint])
    if args.fp16:
        command.append("--fp16")
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if args.freeze_encoder:
        command.append("--freeze-encoder")
    if args.no_augmentation:
        command.append("--no-augmentation")
    else:
        command.extend(["--augmentation-strength", args.augmentation_strength])
    return command


def main() -> None:
    args, extra_args = parse_args()
    for dataset in args.datasets:
        command = build_command(args, dataset=dataset)
        command.extend(extra_args)
        print()
        print(f"=== Fine-tuning TrOCR {SCENARIO_RUNS[dataset]}: {dataset} ===")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
