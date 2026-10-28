"""Configuracao e comandos dos experimentos por seed, sem dependencias de GPU."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/default.yaml"
DEFAULT_MATRIX = PROJECT_ROOT / "configs/experiments/matrix.yaml"
MODEL_SPECS = {
    "cnn": ("train_e1cnn.py", "evaluate_model.py", "e1cnn", "e1cnn_best_checkpoint.pt"),
    "crnn": ("train_e2cnn_bilstm.py", "evaluate_model.py", "e2cnn-bilstm", "e2cnn_bilstm_best_checkpoint.pt"),
    "trocr": ("train_trocr.py", "evaluate_trocr.py", "trocr", "trocr_best"),
    "parseq": ("train_parseq.py", "evaluate_parseq.py", "parseq", "parseq_best"),
}
COMMON_PARAMETERS = (
    "epochs", "batch_size", "learning_rate", "weight_decay", "num_workers",
    "image_height", "image_width", "label_column", "image_column",
)
SEQUENCE_PARAMETERS = (
    "max_target_length", "max_generation_length", "num_beams",
    "gradient_accumulation_steps", "warmup_ratio", "max_grad_norm",
)


def seeded_run_name(name: str, seed: int) -> str:
    """Adiciona a seed uma unica vez e rejeita identificacoes contraditorias."""
    match = re.search(r"_seed(\d+)$", name)
    if match:
        if int(match[1]) != seed:
            raise ValueError(f"Run {name} contradiz --seed {seed}.")
        return name
    return f"{name}_seed{seed}"


def ensure_new_run(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"A pasta {output_dir} ja contem artefatos. Use outro --run-name "
            "ou --output-dir para preservar a execucao existente."
        )


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Esperada exatamente uma linha em {path}.")
    return rows[0]


def as_bool(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"Booleano invalido no registro: {value!r}")
    return value.lower() == "true"


@dataclass(frozen=True)
class Experiment:
    run_name: str
    model: str
    dataset: str

    def name(self, seed: int) -> str:
        return seeded_run_name(self.run_name, seed)

    def evaluation_name(self, seed: int) -> str:
        return f"{self.name(seed)}_evaluation"

    def checkpoint(self, runs_dir: Path, seed: int) -> Path:
        return runs_dir / self.name(seed) / MODEL_SPECS[self.model][3]


@dataclass(frozen=True)
class SeedSettings:
    root: Path
    seeds: tuple[int, ...]
    baseline_seed: int
    experiments: tuple[Experiment, ...]
    runs_dir: Path
    metrics_dir: Path
    protocol: dict = field(default_factory=dict)

    @property
    def additional_seeds(self) -> tuple[int, ...]:
        return tuple(seed for seed in self.seeds if seed != self.baseline_seed)


def load_settings(
    config_path: Path = DEFAULT_CONFIG,
    matrix_path: Path = DEFAULT_MATRIX,
    root: Path = PROJECT_ROOT,
) -> SeedSettings:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    reproducibility = config["reproducibility"]
    seeds = tuple(reproducibility["training_seeds"])
    baseline = reproducibility["baseline_seed"]
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("training_seeds deve conter inteiros entre 0 e 2**32 - 1.")
    if len(set(seeds)) != len(seeds) or baseline not in seeds:
        raise ValueError("Seeds devem ser unicas e incluir baseline_seed.")
    if config["evaluation"]["seed_aggregation"] != "median":
        raise ValueError("A agregacao configurada deve ser median.")
    if config["evaluation"]["require_all_seeds"] is not True:
        raise ValueError("A mediana final requer todas as seeds configuradas.")
    datasets = {"PrescriptionBD": "prescriptionbd", "RxHandBD": "rxhandbd", "combined": "combined"}
    experiments = []
    for entry in matrix["experiments"]:
        name = entry["run_name"]
        if not re.fullmatch(r"e[1-4]\.0[1-3]", name):
            raise ValueError(f"Nome base invalido: {name}")
        if entry["train_dataset"] != entry["test_dataset"]:
            raise ValueError("O executor por seed requer cenarios com treino/teste do mesmo dataset.")
        if entry["model"] not in MODEL_SPECS:
            raise ValueError(f"Modelo desconhecido: {entry['model']}")
        experiments.append(Experiment(name, entry["model"], datasets[entry["train_dataset"]]))
    if not experiments or len({e.run_name for e in experiments}) != len(experiments):
        raise ValueError("A matriz deve conter experimentos com nomes unicos.")
    protocol = config["training"]
    if not protocol.get("protocol_version"):
        raise ValueError("training.protocol_version e obrigatorio.")
    for key in ("batch_size", "gradient_accumulation_steps", "effective_batch_size", "max_epochs"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"training.{key} deve ser inteiro positivo.")
    if not 0 <= protocol["warmup_ratio"] <= 1 or protocol["learning_rate"] <= 0 or protocol["max_grad_norm"] <= 0:
        raise ValueError("Warmup deve estar em [0, 1]; learning_rate e max_grad_norm devem ser positivos.")
    if protocol["batch_size"] * protocol["gradient_accumulation_steps"] != protocol["effective_batch_size"]:
        raise ValueError("Batch efetivo deve ser batch_size * gradient_accumulation_steps.")
    if protocol["validation_population"] != "full" or protocol["checkpoint_metric"] != "val_cer" or protocol["checkpoint_tiebreak"] != "val_accuracy":
        raise ValueError("O protocolo comum exige validacao completa e min CER/desempate max Accuracy.")
    if protocol["optimizer"] != "AdamW" or protocol["scheduler"] != "linear_warmup":
        raise ValueError("O protocolo comum implementa AdamW e linear_warmup.")
    return SeedSettings(
        root, seeds, baseline, tuple(sorted(experiments, key=lambda e: e.run_name)),
        root / config["paths"]["runs_dir"], root / config["paths"]["metrics_dir"], protocol,
    )


def baseline_config(settings: SeedSettings, experiment: Experiment) -> dict[str, str]:
    """Referencia central do protocolo; CSV legado apenas para ferramentas historicas."""
    if settings.protocol:
        return protocol_config(settings, experiment)
    path = settings.runs_dir / experiment.name(settings.baseline_seed) / "run_config.csv"
    row = read_csv_row(path)
    if int(row["seed"]) != settings.baseline_seed:
        raise ValueError(f"Seed de referencia divergente em {path}.")
    for split in ("train", "val"):
        recorded = Path(row[f"{split}_csv"])
        expected = settings.root / "data/splits/scenarios" / experiment.dataset / f"{split}.csv"
        if (settings.root / recorded).resolve() != expected.resolve():
            raise ValueError(f"Split da referencia divergente: {recorded}; esperado {expected}.")
        if not expected.is_file():
            raise FileNotFoundError(expected)
    return row


def protocol_config(settings: SeedSettings, experiment: Experiment) -> dict[str, str]:
    protocol = settings.protocol
    keys = COMMON_PARAMETERS + ("gradient_accumulation_steps", "effective_batch_size", "warmup_ratio",
                                "max_grad_norm", "fp16", "augmentation_enabled", "augmentation_strength",
                                "protocol_version")
    values = {key: protocol["max_epochs"] if key == "epochs" else protocol[key] for key in keys}
    if not protocol["augmentation_enabled"]:
        values["augmentation_strength"] = ""
    values["data_root"] = str(settings.root / "data")
    values["model_kind"] = MODEL_SPECS[experiment.model][2]
    populations = {}
    fingerprints = {}
    for split in ("train", "val", "test"):
        path = settings.root / "data/splits/scenarios" / experiment.dataset / f"{split}.csv"
        fingerprints[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("split", split).lower() == split]
        identifiers = [row["sample_id"] for row in rows]
        if not rows or len(set(identifiers)) != len(rows):
            raise ValueError(f"Populacao vazia ou IDs duplicados em {path}.")
        if any(not row.get(protocol["image_column"]) or not row.get(protocol["label_column"]) for row in rows):
            raise ValueError(f"Imagens/rotulos vazios em {path}.")
        populations[split] = set(identifiers)
        if split != "test":
            values[f"{split}_csv"] = str(path)
            values[f"{split}_samples"] = len(rows)
    if any(populations[a] & populations[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"Sobreposicao de IDs entre splits de {experiment.dataset}.")
    if experiment.model in {"trocr", "parseq"}:
        values.update(protocol["sequence"])
        values.update(protocol[experiment.model])
        values["pretrained"] = protocol["pretrained_transformers"]
    digest_input = {"protocol": protocol, "model": experiment.model, "dataset": experiment.dataset,
                    "split_sha256": fingerprints}
    values["protocol_fingerprint"] = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return {key: str(value) for key, value in values.items()}


def training_command(
    settings: SeedSettings, experiment: Experiment, seed: int,
    reference: dict[str, str], python: str,
) -> list[str]:
    script = MODEL_SPECS[experiment.model][0]
    command = [python, str(settings.root / "src/med_ocr_research/training" / script),
               "--run-name", experiment.name(seed), "--seed", str(seed),
               "--output-dir", str(settings.runs_dir / experiment.name(seed))]
    # O protocolo central e a fonte dos parametros; nenhuma seed treinada e necessaria.
    for key in ("train_csv", "val_csv", "data_root"):
        command.extend([f"--{key.replace('_', '-')}", str(settings.root / reference[key])])
    keys = COMMON_PARAMETERS
    if settings.protocol:
        keys += ("gradient_accumulation_steps", "warmup_ratio", "max_grad_norm",
                 "protocol_version", "protocol_fingerprint")
    if experiment.model in {"trocr", "parseq"}:
        command.extend(["--dataset", experiment.dataset])
        keys += tuple(key for key in SEQUENCE_PARAMETERS if key not in keys)
    if experiment.model == "trocr":
        keys += ("checkpoint", "processor_checkpoint")
        if settings.protocol:
            command.append("--pretrained" if as_bool(reference["pretrained"]) else "--no-pretrained")
    elif experiment.model == "parseq":
        keys += ("model_name", "hub_repo")
        # O registro historico armazena o charset completo, incluindo os extras.
        command.extend(["--extra-characters", reference.get("extra_characters", reference.get("charset", " "))])
        command.append("--pretrained" if as_bool(reference["pretrained"]) else "--no-pretrained")
    for key in keys:
        command.extend([f"--{key.replace('_', '-')}", reference[key]])
    if as_bool(reference["augmentation_enabled"]):
        command.extend(["--augmentation-strength", reference["augmentation_strength"]])
    else:
        command.append("--no-augmentation")
    if settings.protocol or experiment.model in {"trocr", "parseq"}:
        boolean_keys = ("fp16", "gradient_checkpointing") if experiment.model in {"trocr", "parseq"} else ("fp16",)
        for key in boolean_keys:
            flag = key.replace("_", "-")
            command.append(f"--{flag}" if as_bool(reference[key]) else f"--no-{flag}")
        if as_bool(reference.get("freeze_encoder", "False")):
            command.append("--freeze-encoder")
    return command


def training_complete(
    settings: SeedSettings, experiment: Experiment, seed: int, reference: dict[str, str],
) -> bool:
    run_dir = settings.runs_dir / experiment.name(seed)
    if not run_dir.exists():
        return False
    try:
        config = read_csv_row(run_dir / "run_config.csv")
        summary = read_csv_row(run_dir / "run_summary.csv")
    except FileNotFoundError:
        return False
    if config["run_name"] != experiment.name(seed) or int(config["seed"]) != seed:
        raise ValueError(f"Identificacao incorreta em {run_dir}.")
    keys = COMMON_PARAMETERS + ("augmentation_enabled", "augmentation_strength", "train_csv", "val_csv", "data_root")
    if settings.protocol:
        if config.get("protocol_version") != reference["protocol_version"] or config.get("protocol_fingerprint") != reference["protocol_fingerprint"]:
            raise ValueError(f"{run_dir}: protocolo incompativel; arquive antes de refazer o treinamento.")
        keys += ("gradient_accumulation_steps", "effective_batch_size", "warmup_ratio", "max_grad_norm",
                 "fp16", "train_samples", "val_samples")
    if experiment.model in {"trocr", "parseq"}:
        keys += SEQUENCE_PARAMETERS + ("fp16", "gradient_checkpointing", "freeze_encoder")
    keys += {"trocr": ("checkpoint", "processor_checkpoint"),
             "parseq": ("model_name", "hub_repo", "pretrained")}.get(experiment.model, ())
    if settings.protocol and experiment.model == "trocr":
        keys += ("pretrained",)
    if not settings.protocol and experiment.model == "parseq":
        keys += ("charset",)
    for key in keys:
        actual, expected = config[key], reference[key]
        if key in {"train_csv", "val_csv", "data_root"}:
            actual = str((settings.root / actual).resolve())
            expected = str((settings.root / expected).resolve())
        if actual != expected:
            raise ValueError(f"{run_dir}: {key} difere da referencia ({actual!r} != {expected!r}).")
    checkpoint = experiment.checkpoint(settings.runs_dir, seed)
    weights_present = checkpoint.is_file()
    if checkpoint.is_dir():
        weights_present = any(checkpoint.glob("*.pt")) or any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("pytorch_model*.bin"))
    return int(summary["epochs_completed"]) == int(reference["epochs"]) and weights_present


def evaluation_command(
    settings: SeedSettings, experiment: Experiment, seed: int, reference: dict[str, str], python: str,
) -> list[str]:
    _, script, model_kind, _ = MODEL_SPECS[experiment.model]
    test_csv = settings.root / "data/splits/scenarios" / experiment.dataset / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(test_csv)
    command = [python, str(settings.root / "src/med_ocr_research/evaluation" / script),
               "--checkpoint", str(experiment.checkpoint(settings.runs_dir, seed)),
               "--test-csv", str(test_csv), "--data-root", str(settings.root / reference["data_root"]),
               "--run-name", experiment.evaluation_name(seed), "--output-dir", str(settings.metrics_dir)]
    if experiment.model in {"cnn", "crnn"}:
        command.extend(["--model-kind", model_kind])
    for key in ("label_column", "image_column", "image_height", "image_width"):
        command.extend([f"--{key.replace('_', '-')}", reference[key]])
    if experiment.model in {"trocr", "parseq"}:
        for key in ("max_generation_length", "num_beams"):
            command.extend([f"--{key.replace('_', '-')}", reference[key]])
    return command
