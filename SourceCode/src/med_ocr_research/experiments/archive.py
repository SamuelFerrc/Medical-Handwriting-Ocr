"""Preserva artefatos de protocolos antigos fora da comparacao ativa."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

from med_ocr_research.experiments.progress import runtime_directory
from med_ocr_research.experiments.seeds import SeedSettings, baseline_config, read_csv_row


def incompatible_artifacts(settings: SeedSettings) -> list[tuple[Path, str]]:
    artifacts = []
    for experiment in settings.experiments:
        expected = baseline_config(settings, experiment)
        for seed in settings.seeds:
            run_dir = settings.runs_dir / experiment.name(seed)
            metrics_dir = settings.metrics_dir / experiment.evaluation_name(seed)
            if not run_dir.exists() and not metrics_dir.exists():
                continue
            try:
                recorded = read_csv_row(run_dir / "run_config.csv")
            except (FileNotFoundError, ValueError):
                recorded = {}
            if (recorded.get("protocol_version") == expected["protocol_version"]
                    and recorded.get("protocol_fingerprint") == expected["protocol_fingerprint"]):
                continue
            reason = f"Protocolo {recorded.get('protocol_version', 'legado sem identificacao')} incompativel com {expected['protocol_version']}"
            for path in (run_dir, metrics_dir):
                if path.exists():
                    artifacts.append((path, reason))
    if artifacts:
        for path in (settings.metrics_dir / "seed_summary", runtime_directory(settings.root) / "pipeline.json",
                     runtime_directory(settings.root) / "training.log"):
            if path.exists():
                artifacts.append((path, "Resumo/estado/log anterior a mudanca de protocolo"))
    return artifacts


def archive_incompatible_runs(settings: SeedSettings, *, dry_run: bool = False) -> dict:
    directory = runtime_directory(settings.root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Ha treinamento em execucao. Interrompa-o antes de arquivar artefatos.") from None
        candidates = incompatible_artifacts(settings)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = settings.root / "outputs/archive" / timestamp
        entries = [{"source": str(path.relative_to(settings.root)),
                    "archived": str((destination / path.relative_to(settings.root / 'outputs')).relative_to(settings.root)),
                    "reason": reason} for path, reason in candidates]
        manifest = {"archived_at_utc": timestamp, "new_protocol": settings.protocol["protocol_version"],
                    "artifact_contents": "preserved; original internal paths are historical",
                    "entries": entries}
        if dry_run or not entries:
            return manifest
        destination.mkdir(parents=True, exist_ok=False)
        applied = []
        try:
            for entry in entries:
                source, target = settings.root / entry["source"], settings.root / entry["archived"]
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                applied.append((source, target))
            (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            for source, target in reversed(applied):
                target.rename(source)
            raise
        manifest["manifest_path"] = str(destination / "manifest.json")
        return manifest
