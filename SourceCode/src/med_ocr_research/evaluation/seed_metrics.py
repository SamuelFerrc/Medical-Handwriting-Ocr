"""Medianas entre treinamentos independentes sobre o mesmo conjunto de teste."""
from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from med_ocr_research.evaluation.provenance import checkpoint_fingerprint
from med_ocr_research.experiments.seeds import (
    Experiment, SeedSettings, baseline_config, read_csv_row, training_complete,
)


METRICS = ("exact_match_accuracy", "cer", "wer")


def test_population(path: Path, label_column: str, *, predictions: bool = False) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not predictions:
        rows = [row for row in rows if row.get("split", "test").lower() == "test"]
    population = sorted((row["sample_id"], row[label_column]) for row in rows)
    if not population or len({key for key, _ in population}) != len(population):
        raise ValueError(f"Amostras vazias ou sample_id repetido em {path}.")
    return population


def read_seed_metrics(settings: SeedSettings, experiment: Experiment, seed: int) -> dict[str, Any]:
    directory = settings.metrics_dir / experiment.evaluation_name(seed)
    path = directory / "metrics.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    config = read_csv_row(settings.runs_dir / experiment.name(seed) / "run_config.csv")
    if config["run_name"] != experiment.name(seed) or int(config["seed"]) != seed:
        raise ValueError(f"Seed/nome da run incorretos em {path}.")
    if summary["run_name"] != experiment.evaluation_name(seed):
        raise ValueError(f"Nome da avaliacao incorreto em {path}.")
    expected_source = experiment.checkpoint(settings.runs_dir, seed).resolve()
    if (settings.root / summary["source"]).resolve() != expected_source:
        raise ValueError(f"Checkpoint incorreto em {path}.")
    provenance = {}
    if settings.protocol:
        reference = baseline_config(settings, experiment)
        for key in ("protocol_version", "protocol_fingerprint"):
            if config.get(key) != reference[key] or summary.get(key) != reference[key]:
                raise ValueError(f"Protocolo da avaliacao incompativel ({key}) em {path}.")
        if summary.get("training_seed") != seed:
            raise ValueError(f"Seed da avaliacao incorreta em {path}.")
        digest = checkpoint_fingerprint(expected_source)
        if summary.get("checkpoint_fingerprint") != digest:
            raise ValueError(f"Pesos do checkpoint mudaram ou nao foram identificados em {path}; refaca a avaliacao.")
        provenance = {"protocol_version": reference["protocol_version"],
                      "protocol_fingerprint": reference["protocol_fingerprint"],
                      "checkpoint_fingerprint": digest}
    expected_test = settings.root / "data/splits/scenarios" / experiment.dataset / "test.csv"
    if not summary.get("test_csv") or (settings.root / summary["test_csv"]).resolve() != expected_test.resolve():
        raise ValueError(f"Split de teste incorreto em {path}.")
    population = test_population(expected_test, config["label_column"])
    actual = test_population(directory / "evaluated_predictions.csv", "text_true", predictions=True)
    if actual != population or summary["metrics"]["samples"] != len(population):
        raise ValueError(f"A avaliacao {path} nao cobre exatamente o teste completo com os mesmos rotulos.")
    values = {metric: float(summary["metrics"][metric]) for metric in METRICS}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"Metricas nao finitas ou negativas em {path}.")
    if values["exact_match_accuracy"] > 1:
        raise ValueError(f"Accuracy fora de [0, 1] em {path}.")
    return {
        "base_run_name": experiment.run_name, "model": experiment.model,
        "dataset": experiment.dataset, "seed": seed, "samples": len(population),
        **values, **provenance, "metrics_path": str(path),
    }


def aggregate_seed_metrics(settings: SeedSettings) -> dict[str, Any]:
    missing = [experiment.evaluation_name(seed)
               for experiment in settings.experiments for seed in settings.seeds
               if not (settings.metrics_dir / experiment.evaluation_name(seed) / "metrics.json").is_file()]
    if missing:
        raise FileNotFoundError(
            "Mediana final indisponivel: faltam as seguintes avaliacoes das seeds configuradas:\n  "
            + "\n  ".join(missing)
        )
    individual, medians = [], []
    for experiment in settings.experiments:
        reference = baseline_config(settings, experiment)
        for seed in settings.seeds:
            if not training_complete(settings, experiment, seed, reference):
                raise ValueError(f"Treinamento incompleto: {experiment.name(seed)}")
        rows = [read_seed_metrics(settings, experiment, seed) for seed in settings.seeds]
        individual.extend(rows)
        medians.append({
            "base_run_name": experiment.run_name, "model": experiment.model,
            "dataset": experiment.dataset, "seeds": list(settings.seeds),
            "seed_count": len(rows), "samples_per_seed": rows[0]["samples"],
            **{f"{metric}_median": statistics.median(row[metric] for row in rows) for metric in METRICS},
        })
    return {
        "aggregation": "median", "training_seeds": list(settings.seeds),
        "protocol_version": settings.protocol.get("protocol_version", "historico"),
        "training_protocol": settings.protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_seed": individual, "medians": medians,
    }


def write_seed_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (("per_seed", "per_seed_metrics.csv"), ("medians", "seed_medians.csv")):
        rows = summary[key]
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "seed_medians.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
