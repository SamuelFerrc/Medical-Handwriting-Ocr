"""Painel HTTP de leitura dos artefatos, independente do processo de treino."""
from __future__ import annotations

import csv
import hmac
import json
import math
import os
import secrets
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import yaml

from med_ocr_research.experiments.progress import pipeline_status, read_json, runtime_directory, utc_now
from med_ocr_research.experiments.seeds import SeedSettings, baseline_config, training_complete


_REFERENCE_CACHE = OrderedDict()
_REFERENCE_CACHE_LOCK = Lock()
_REFERENCE_CACHE_LIMIT = 128


def expected_reference(settings, experiment):
    """Cache CSV validation until configuration or any split file changes."""
    split_stats = []
    for split in ("train", "val", "test"):
        path = settings.root / "data/splits/scenarios" / experiment.dataset / f"{split}.csv"
        stat = path.stat()
        split_stats.append((stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns))
    key = (str(settings.root.resolve()), json.dumps(settings.protocol, sort_keys=True),
           experiment, tuple(split_stats))
    # Serialize a cold cache miss so concurrent HTTP polls do not all parse the
    # same population while an earlier request is still preparing the response.
    with _REFERENCE_CACHE_LOCK:
        if key not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[key] = baseline_config(settings, experiment)
            while len(_REFERENCE_CACHE) > _REFERENCE_CACHE_LIMIT:
                _REFERENCE_CACHE.popitem(last=False)
        _REFERENCE_CACHE.move_to_end(key)
        return dict(_REFERENCE_CACHE[key])


def monitor_address(config_path: Path) -> tuple[str, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monitor = config.get("monitoring", {})
    return str(monitor.get("host", "127.0.0.1")), int(monitor.get("port", 8765))


def access_token(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "access_token"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise ValueError(f"Token de acesso invalido em {path}.")
        return token
    token = secrets.token_urlsafe(32)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(token + "\n")
    return token


def csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, ValueError, csv.Error):
        return []


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def log_tail(path: Path, limit: int = 48_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read(limit)
        return "\n".join(data.decode("utf-8", errors="replace").replace("\r", "\n").splitlines()[-160:])
    except OSError:
        return "O log aparecera aqui quando o treinamento for iniciado pelo servico."


def checkpoint_predates_evaluation(checkpoint: Path, evaluation: dict) -> bool:
    """Cheap monitor freshness check; final aggregation also hashes the weights."""
    try:
        evaluated_at = datetime.fromisoformat(evaluation["created_at_utc"])
        if evaluated_at.tzinfo is None:
            return False
        if checkpoint.is_file():
            weights = [checkpoint]
        elif checkpoint.is_dir():
            ignored = ("optimizer", "scheduler", "scaler", "rng_state", "training_args", "tokenizer")
            weights = [path for pattern in ("model*.safetensors", "pytorch_model*.bin", "*.pt")
                       for path in checkpoint.rglob(pattern)
                       if path.is_file() and not path.name.startswith(ignored)]
        else:
            return False
        return bool(weights) and all(path.stat().st_mtime <= evaluated_at.timestamp() for path in weights)
    except (KeyError, TypeError, ValueError, OSError):
        return False


def evaluation_is_current(settings, experiment, seed, expected, config, evaluation) -> bool:
    if any(config.get(key) != expected.get(key) or evaluation.get(key) != expected.get(key)
           for key in ("protocol_version", "protocol_fingerprint")):
        return False
    if evaluation.get("training_seed") != seed or not evaluation.get("checkpoint_fingerprint"):
        return False
    if evaluation.get("run_name") != experiment.evaluation_name(seed):
        return False
    try:
        checkpoint = experiment.checkpoint(settings.runs_dir, seed)
        if (settings.root / evaluation["source"]).resolve() != checkpoint.resolve():
            return False
        if not checkpoint_predates_evaluation(checkpoint, evaluation):
            return False
        metrics = evaluation["metrics"]
        for key in ("exact_match_accuracy", "cer", "wer"):
            value = number(metrics.get(key))
            if value is None or value < 0 or (key == "exact_match_accuracy" and value > 1):
                return False
        return number(metrics.get("samples")) is not None and metrics["samples"] > 0
    except (KeyError, TypeError, ValueError):
        return False


def build_snapshot(settings: SeedSettings) -> dict:
    directory = runtime_directory(settings.root)
    state = pipeline_status(directory)
    runs = []
    current_evaluations = {}
    for experiment in settings.experiments:
        expected = None
        if settings.protocol:
            try:
                expected = expected_reference(settings, experiment)
            except (OSError, ValueError, KeyError):
                pass
        for seed in settings.seeds:
            name = experiment.name(seed)
            run_dir = settings.runs_dir / name
            config = next(iter(csv_rows(run_dir / "run_config.csv")), {})
            summary = next(iter(csv_rows(run_dir / "run_summary.csv")), {})
            history = [row for row in csv_rows(run_dir / "training_history.csv") if number(row.get("epoch")) is not None]
            latest = history[-1] if history else {}
            evaluation = read_json(settings.metrics_dir / experiment.evaluation_name(seed) / "metrics.json")
            metrics = evaluation.get("metrics", {})
            epochs = number(config.get("epochs"))
            completed = number(summary.get("epochs_completed"))
            trained = epochs is not None and completed == epochs
            if settings.protocol:
                try:
                    trained = bool(expected) and training_complete(settings, experiment, seed, expected)
                except (OSError, ValueError, KeyError, TypeError):
                    trained = False
                if not trained or not evaluation_is_current(settings, experiment, seed, expected, config, evaluation):
                    metrics = {}
            if metrics:
                current_evaluations[(experiment.run_name, seed)] = evaluation
            status = "evaluated" if metrics else "trained" if trained else "partial" if run_dir.exists() else "pending"
            if state.get("current_run") == name and state.get("status") in {"running", "failed", "stopped", "interrupted"}:
                status = state["phase"] if state["status"] == "running" else state["status"]
            runs.append({
                "name": name, "model": experiment.model, "dataset": experiment.dataset, "seed": seed,
                "status": status, "epoch": number(latest.get("epoch")) or completed or 0,
                "epochs": epochs, "trained": trained, "evaluated": bool(metrics),
                "val_loss": number(latest.get("val_loss")), "val_accuracy": number(latest.get("val_accuracy")),
                "val_cer": number(latest.get("val_cer")), "val_wer": number(latest.get("val_wer")),
                "metrics": {key: number(metrics.get(key)) for key in ("exact_match_accuracy", "cer", "wer")},
            })
    final = read_json(settings.metrics_dir / "seed_summary/seed_medians.json")
    medians = final.get("medians", []) if final.get("training_seeds") == list(settings.seeds) else []
    if settings.protocol:
        expected_count = len(settings.experiments) * len(settings.seeds)
        rows = final.get("per_seed", [])
        valid = (final.get("protocol_version") == settings.protocol["protocol_version"]
                 and final.get("training_protocol") == settings.protocol
                 and len(current_evaluations) == expected_count and len(rows) == expected_count)
        seen = set()
        for row in rows:
            key = (row.get("base_run_name"), row.get("seed"))
            current = current_evaluations.get(key, {})
            if key in seen or not current:
                valid = False
                break
            seen.add(key)
            if any(row.get(field) != current.get(field) for field in (
                "protocol_version", "protocol_fingerprint", "checkpoint_fingerprint",
            )) or any(number(row.get(metric)) != number(current["metrics"].get(metric)) for metric in (
                "exact_match_accuracy", "cer", "wer", "samples",
            )):
                valid = False
                break
        if not valid:
            medians = []
    return {"updated_at": utc_now(), "pipeline": state, "seeds": list(settings.seeds), "runs": runs,
            "protocol_version": settings.protocol.get("protocol_version", "historico"),
            "trained": sum(row["trained"] for row in runs), "evaluated": sum(row["evaluated"] for row in runs),
            "medians": medians, "log": log_tail(directory / "training.log")}


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], settings: SeedSettings, token: str):
        self.settings = settings
        self.token = token
        super().__init__(address, MonitorHandler)


class MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args) -> None:
        # Requisicoes periodicas nao enchem o log; o token nunca entra em URLs do servidor.
        pass

    def send(self, code: int, content: bytes, content_type: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # The browser may leave or cancel a poll while a response is built.
            self.close_connection = True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.send(200, Path(__file__).with_name("monitor.html").read_bytes(), "text/html; charset=utf-8")
            return
        credential = self.headers.get("Authorization", "")
        if not hmac.compare_digest(credential.encode(), f"Bearer {self.server.token}".encode()):
            self.send(401, b'{"error":"Token de acesso necessario."}', "application/json")
            return
        if path == "/api/status":
            data = build_snapshot(self.server.settings)
            self.send(200, json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")
        elif path == "/api/medians.csv":
            # Apenas o resumo final; nao e um servidor de arquivos do projeto.
            target = self.server.settings.metrics_dir / "seed_summary/seed_medians.csv"
            if target.is_file() and build_snapshot(self.server.settings)["medians"]:
                self.send(200, target.read_bytes(), "text/csv; charset=utf-8")
            else:
                self.send(404, b"Medianas ainda indisponiveis.", "text/plain; charset=utf-8")
        else:
            self.send(404, b"Nao encontrado.", "text/plain; charset=utf-8")
