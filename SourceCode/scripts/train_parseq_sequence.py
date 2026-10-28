from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from med_ocr_research.experiments.seeds import seeded_run_name

TRAIN_SCRIPT = PROJECT_ROOT / "src/med_ocr_research/training/train_parseq.py"
SCENARIO_RUNS = {
    "prescriptionbd": "e4.01",
    "rxhandbd": "e4.02",
    "combined": "e4.03",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Faz fine-tuning do PARSeq em sequencia. Por padrao roda "
            "prescriptionbd, rxhandbd e combined."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SCENARIO_RUNS),
        default=list(SCENARIO_RUNS),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--model-name", default="parseq")
    parser.add_argument(
        "--hub-repo",
        default="baudm/parseq:1902db043c029a7e03a3818c616c06600af574be",
    )
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--extra-characters", default=" ")
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-generation-length", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
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
    parser.add_argument(
        "--augmentation-strength",
        choices=("light", "medium", "strong"),
        default="strong",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def build_command(args: argparse.Namespace, dataset: str) -> list[str]:
    split_dir = PROJECT_ROOT / "data/splits/scenarios" / dataset
    missing = [
        path
        for path in (split_dir / "train.csv", split_dir / "val.csv")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Split ausente para {dataset}: {', '.join(map(str, missing))}"
        )

    command = [
        args.python,
        str(TRAIN_SCRIPT),
        "--run-name",
        seeded_run_name(SCENARIO_RUNS[dataset], args.seed),
        "--dataset",
        dataset,
        "--data-root",
        str(args.data_root),
        "--model-name",
        args.model_name,
        "--hub-repo",
        args.hub_repo,
        "--extra-characters",
        args.extra_characters,
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
    command.append("--pretrained" if args.pretrained else "--no-pretrained")
    command.append("--fp16" if args.fp16 else "--no-fp16")
    command.append(
        "--gradient-checkpointing"
        if args.gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
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
        command = build_command(args, dataset)
        command.extend(extra_args)
        print()
        print(f"=== Fine-tuning PARSeq {SCENARIO_RUNS[dataset]}: {dataset} ===")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
