"""Executa todas as seeds pendentes do protocolo comum e consolida as medianas."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_ocr_research.evaluation.seed_metrics import aggregate_seed_metrics, read_seed_metrics, write_seed_summary
from med_ocr_research.experiments.progress import PipelineProgress, runtime_directory
from med_ocr_research.experiments.seeds import (
    DEFAULT_CONFIG, DEFAULT_MATRIX, baseline_config, evaluation_command,
    load_settings, training_command, training_complete,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Valida e mostra o plano sem treinar nem gravar arquivos.")
    args = parser.parse_args()
    settings = load_settings(args.config, args.matrix)
    progress = nullcontext(None) if args.dry_run else PipelineProgress(runtime_directory(settings.root), settings.seeds)
    with progress as tracker:
        execute(args, settings, tracker)


def execute(args, settings, tracker=None) -> None:
    jobs = []
    # Valida toda a matriz antes de iniciar o primeiro treinamento.
    for experiment in settings.experiments:
        reference = baseline_config(settings, experiment)
        for seed in settings.seeds:
            complete = training_complete(settings, experiment, seed, reference)
            run_dir = settings.runs_dir / experiment.name(seed)
            if not complete and run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(
                    f"Execucao incompleta em {run_dir}. Preserve-a em outra pasta antes de repetir "
                    "o treinamento; este executor nao retoma pesos/otimizador de treinos parciais."
                )
            train = None if complete else training_command(settings, experiment, seed, reference, args.python)
            metric_dir = settings.metrics_dir / experiment.evaluation_name(seed)
            evaluate = evaluation_command(settings, experiment, seed, reference, args.python)
            if (metric_dir / "metrics.json").exists() and complete:
                try:
                    read_seed_metrics(settings, experiment, seed)
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    print(f"Avaliacao sera refeita para {experiment.name(seed)}: {exc}", flush=True)
                else:
                    evaluate = None
            jobs.append((experiment.name(seed), train, evaluate))

    print(f"Seeds: {list(settings.seeds)}; protocolo: {settings.protocol.get('protocol_version', 'historico')}", flush=True)
    print(f"Treinamentos pendentes: {sum(train is not None for _, train, _ in jobs)}; "
          f"avaliacoes pendentes: {sum(evaluate is not None for _, _, evaluate in jobs)}", flush=True)
    if tracker:
        tracker.update(status="running", total_steps=sum(command is not None for _, train, evaluate in jobs for command in (train, evaluate)))
    for name, train, evaluate in jobs:
        print(f"\n{name}", flush=True)
        if train is None:
            print("  Treinamento completo preservado.", flush=True)
        for phase, command in (("training", train), ("evaluation", evaluate)):
            if command is not None:
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    if tracker:
                        tracker.update(phase=phase, current_run=name)
                    subprocess.run(command, cwd=settings.root, check=True,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    if tracker:
                        tracker.update(finished_steps=tracker.state["finished_steps"] + 1)
    if args.dry_run:
        print("Ao concluir as avaliacoes, serao calculadas as medianas das tres metricas entre todas as seeds.")
    else:
        if tracker:
            tracker.update(status="aggregating", phase="aggregating", current_run=None)
        summary = aggregate_seed_metrics(settings)
        destination = settings.metrics_dir / "seed_summary"
        write_seed_summary(summary, destination)
        print(f"Medianas e resultados individuais salvos em {destination}")


if __name__ == "__main__":
    main()
