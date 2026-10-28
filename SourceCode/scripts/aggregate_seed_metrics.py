from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_ocr_research.evaluation.seed_metrics import aggregate_seed_metrics, write_seed_summary
from med_ocr_research.experiments.seeds import DEFAULT_CONFIG, DEFAULT_MATRIX, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida medianas por modelo/cenario, exigindo todas as seeds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()
    settings = load_settings(args.config, args.matrix)
    try:
        summary = aggregate_seed_metrics(settings)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    destination = settings.metrics_dir / "seed_summary"
    write_seed_summary(summary, destination)
    print(f"Medianas e resultados individuais salvos em {destination}")


if __name__ == "__main__":
    main()
