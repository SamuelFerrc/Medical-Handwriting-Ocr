from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_ocr_research.experiments.archive import archive_incompatible_runs
from med_ocr_research.experiments.seeds import DEFAULT_CONFIG, DEFAULT_MATRIX, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Arquiva resultados de protocolos antigos preservando os arquivos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = archive_incompatible_runs(load_settings(args.config, args.matrix), dry_run=args.dry_run)
    for entry in manifest["entries"]:
        print(f"{entry['source']} -> {entry['archived']} ({entry['reason']})")
    print(f"{len(manifest['entries'])} artefatos {'planejados' if args.dry_run else 'arquivados'}.")
    if manifest.get("manifest_path"):
        print(f"Manifesto: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
