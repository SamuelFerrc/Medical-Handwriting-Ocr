from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_ocr_research.experiments.migration import migrate_seed_runs
from med_ocr_research.experiments.seeds import DEFAULT_CONFIG, DEFAULT_MATRIX, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica runs e avaliacoes existentes pela seed registrada.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config, args.matrix)
    moves = migrate_seed_runs(settings, dry_run=args.dry_run)
    for source, destination in moves:
        print(f"{source.relative_to(PROJECT_ROOT)} -> {destination.relative_to(PROJECT_ROOT)}")
    print(f"{len(moves)} pastas {'planejadas' if args.dry_run else 'renomeadas'}; pesos preservados.")


if __name__ == "__main__":
    main()
