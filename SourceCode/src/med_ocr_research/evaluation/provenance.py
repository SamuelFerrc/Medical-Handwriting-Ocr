"""Training/checkpoint identity for rejecting stale evaluation results."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


def _update_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)


def checkpoint_fingerprint(source: str | Path) -> str:
    """Hash checkpoint bytes without loading torch or trusting timestamps.

    A file uses its ordinary SHA256. A model bundle hashes sorted relative
    weight filenames, sizes and contents; tokenizer/config and optimizer state
    files do not participate. Moving the bundle does not change its identity.
    """
    source = Path(source)
    digest = hashlib.sha256()
    if source.is_file():
        _update_from_file(digest, source)
        return digest.hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint ausente: {source}")

    ignored_prefixes = ("optimizer", "scheduler", "scaler", "rng_state", "training_args", "tokenizer")
    weights = sorted(
        {
            path
            for pattern in ("model*.safetensors", "pytorch_model*.bin", "*.pt")
            for path in source.rglob(pattern)
            if path.is_file() and not path.name.startswith(ignored_prefixes)
        },
        key=lambda path: path.relative_to(source).as_posix(),
    )
    if not weights:
        raise FileNotFoundError(f"Nenhum arquivo de pesos encontrado em {source}")
    digest.update(b"med-ocr-checkpoint-bundle-v1\0")
    for path in weights:
        name = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        _update_from_file(digest, path)
    return digest.hexdigest()


def training_provenance(checkpoint: str | Path | None) -> dict[str, str | int]:
    """Read the adjacent run configuration when evaluating a trained model.

    Standalone prediction CSVs and checkpoints without a run configuration
    remain valid evaluation inputs, without inventing protocol metadata.
    """
    if checkpoint is None:
        return {}
    source = Path(checkpoint)
    candidates = [source.parent / "run_config.csv"]
    if source.is_dir():
        candidates.insert(0, source / "run_config.csv")
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        return {}
    with config_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Configuracao deve conter exatamente uma execucao: {config_path}")
    config = rows[0]
    metadata: dict[str, str | int] = {}
    for key in ("protocol_version", "protocol_fingerprint"):
        if config.get(key):
            metadata[key] = config[key]
    if config.get("seed"):
        metadata["training_seed"] = int(config["seed"])
    metadata["checkpoint_fingerprint"] = checkpoint_fingerprint(source)
    return metadata
