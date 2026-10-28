"""Renomeia artefatos legados sem desserializar ou regravar os pesos."""
from __future__ import annotations

import re
from pathlib import Path

from med_ocr_research.experiments.seeds import SeedSettings, read_csv_row


def migration_plan(settings: SeedSettings) -> list[tuple[Path, Path]]:
    moves = []
    for experiment in settings.experiments:
        source = settings.runs_dir / experiment.run_name
        destination = settings.runs_dir / experiment.name(settings.baseline_seed)
        if source.exists() and destination.exists():
            raise FileExistsError(f"Origem e destino existem: {source}, {destination}")
        current = source if source.exists() else destination
        row = read_csv_row(current / "run_config.csv")
        if int(row["seed"]) != settings.baseline_seed:
            raise ValueError(f"A seed registrada em {current} nao e {settings.baseline_seed}.")
        if source.exists():
            moves.append((source, destination))

        suffix = {"cnn": "cnn_evaluation", "crnn": "_e2cnn_bilstm_evaluation",
                  "trocr": "_trocr_evaluation", "parseq": "_parseq_evaluation"}[experiment.model]
        old_metrics = settings.metrics_dir / f"{experiment.run_name}{suffix}"
        new_metrics = settings.metrics_dir / experiment.evaluation_name(settings.baseline_seed)
        if old_metrics.exists():
            if new_metrics.exists():
                raise FileExistsError(f"Origem e destino existem: {old_metrics}, {new_metrics}")
            moves.append((old_metrics, new_metrics))
    return moves


def migrate_seed_runs(settings: SeedSettings, *, dry_run: bool = False) -> list[tuple[Path, Path]]:
    moves = migration_plan(settings)
    if dry_run or not moves:
        return moves
    replacements = {source.name: destination.name for source, destination in moves}
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])(?:" + "|".join(re.escape(name) for name in sorted(replacements, key=len, reverse=True))
        + r")(?![A-Za-z0-9_.-])"
    )
    # Apenas os arquivos com identificadores/caminhos; predicoes, historicos e pesos
    # sao preservados byte a byte. Checkpoints .pt conservam seus metadados historicos.
    metadata_names = {"run_config.csv", "run_summary.csv", "training_metadata.json", "metrics.json", "evaluation.md"}
    edits = []
    for source, _ in moves:
        for path in source.rglob("*"):
            if path.is_file() and path.name in metadata_names:
                original = path.read_bytes()
                updated = pattern.sub(lambda match: replacements[match[0]], original.decode("utf-8")).encode("utf-8")
                if updated != original:
                    edits.append((path, original, updated))
    applied_edits, applied_moves = [], []
    try:
        for path, original, updated in edits:
            applied_edits.append((path, original))
            path.write_bytes(updated)
        for source, destination in moves:
            source.rename(destination)
            applied_moves.append((source, destination))
    except Exception:
        for source, destination in reversed(applied_moves):
            destination.rename(source)
        for path, original in reversed(applied_edits):
            path.write_bytes(original)
        raise
    return moves
