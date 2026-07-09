from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.evaluation.metrics import (  # noqa: E402
    aggregate_metrics,
    label_distribution,
    levenshtein_distance,
    sequence_edit_distance,
)


DEFAULT_TEST_CSV = PROJECT_ROOT / "data/splits/scenarios/combined/test.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs/metrics"
CTC_TARGET = "ctc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia predicoes OCR/HTR com Exact Match Accuracy, CER e WER. "
            "Pode receber um CSV de predicoes ou gerar predicoes a partir de um .pt."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--predictions-csv",
        type=Path,
        help=(
            "CSV ja contendo predicoes. Esperado: text_true/text_pred ou "
            "<label-column>/<prediction-column>."
        ),
    )
    input_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint .pt usado para gerar predicoes antes da avaliacao.",
    )

    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--model-kind",
        choices=("e1cnn", "e2cnn-bilstm", "torchscript-classifier"),
        default="e1cnn",
        help=(
            "Adaptador usado quando --checkpoint e informado. Para novos modelos, "
            "gere um CSV no contrato universal ou adicione um adaptador aqui."
        ),
    )
    parser.add_argument(
        "--class-map-json",
        type=Path,
        default=None,
        help="Mapa label->indice para modelos TorchScript classificadores.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Funcao Python module:function para gerar predicoes de qualquer .pt. "
            "A funcao recebe argparse.Namespace e deve retornar um DataFrame com "
            "text_true e text_pred."
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument(
        "--predictions-output-dir",
        type=Path,
        default=None,
        help=(
            "Diretorio opcional para salvar predicoes avaliadas. "
            "Por padrao, usa a pasta da run em outputs/metrics/<run-name>/."
        ),
    )
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--prediction-column", default="text_pred")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def infer_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return sanitize_name(args.run_name)
    source = args.predictions_csv or args.checkpoint
    if source is None:
        return "evaluation"
    return sanitize_name(source.stem)


def sanitize_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    safe = safe.strip("._-")
    return safe or "evaluation"


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(str(image_path))
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    return data_root / path


def load_image_tensor(
    image_path: Path,
    image_size: tuple[int, int],
    device: Any,
) -> Any:
    import torch

    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize((image_size[1], image_size[0]))
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)


def is_ctc_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return checkpoint.get("target_kind") == CTC_TARGET


def load_class_maps(
    checkpoint: dict[str, Any],
    class_map_json: Path | None,
) -> tuple[dict[str, int], dict[int, str]]:
    if is_ctc_checkpoint(checkpoint) and "char_to_idx" in checkpoint:
        class_to_idx = {
            str(label): int(index)
            for label, index in checkpoint["char_to_idx"].items()
        }
    elif "class_to_idx" in checkpoint:
        class_to_idx = {
            str(label): int(index)
            for label, index in checkpoint["class_to_idx"].items()
        }
    elif class_map_json is not None:
        class_to_idx = {
            str(label): int(index)
            for label, index in json.loads(class_map_json.read_text(encoding="utf-8")).items()
        }
    else:
        raise ValueError(
            "Nao foi encontrado class_to_idx no checkpoint. Informe --class-map-json."
        )
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def generate_predictions_from_checkpoint(args: argparse.Namespace) -> pd.DataFrame:
    if args.adapter is not None:
        return generate_predictions_from_custom_adapter(args)

    import torch

    device = choose_device(args.device)
    checkpoint_obj = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint = checkpoint_obj if isinstance(checkpoint_obj, dict) else {}
    class_to_idx, idx_to_class = load_class_maps(checkpoint, args.class_map_json)
    uses_ctc = is_ctc_checkpoint(checkpoint)

    image_size = infer_image_size(args, checkpoint)
    model = load_model(args, checkpoint, class_to_idx, device)
    model.eval()

    rows = pd.read_csv(args.test_csv)
    if args.split_column in rows.columns:
        rows = rows[rows[args.split_column].astype(str).str.lower() == "test"].copy()
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    required = {args.image_column, args.label_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV de teste: {sorted(missing)}")

    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row_index, row in rows.reset_index(drop=True).iterrows():
            image_path = resolve_image_path(args.data_root, row[args.image_column])
            image_tensor = load_image_tensor(image_path, image_size, device)
            logits = model(image_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if uses_ctc:
                decoded = model.ctc_decode(
                    logits,
                    idx_to_token=idx_to_class,
                    blank_idx=int(checkpoint.get("blank_idx", 0)),
                )
                predicted_text = str(decoded[0])
                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities.max(dim=-1).values.mean()
            else:
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_idx = probabilities.max(dim=1)
                predicted_text = idx_to_class[int(predicted_idx.item())]

            prediction_rows.append(
                {
                    "sample_id": row.get(args.sample_id_column, row_index),
                    "dataset": row.get(args.dataset_column, ""),
                    "split": row.get(args.split_column, "test"),
                    "image_path": row.get(args.image_column, ""),
                    "text_true": str(row[args.label_column]),
                    "text_pred": str(predicted_text),
                    "confidence": float(confidence.item()),
                }
            )

    return pd.DataFrame(prediction_rows)


def generate_predictions_from_custom_adapter(args: argparse.Namespace) -> pd.DataFrame:
    if ":" not in args.adapter:
        raise ValueError("--adapter deve estar no formato module:function.")

    module_name, function_name = args.adapter.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    predict_function = getattr(module, function_name)
    predictions = predict_function(args)
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("O adapter precisa retornar um pandas.DataFrame.")

    required = {"text_true", "text_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no retorno do adapter: {sorted(missing)}")
    return predictions


def choose_device(device_arg: str) -> Any:
    import torch

    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_image_size(args: argparse.Namespace, checkpoint: dict[str, Any]) -> tuple[int, int]:
    if args.image_height is not None and args.image_width is not None:
        return (args.image_height, args.image_width)
    if "image_size" in checkpoint:
        height, width = checkpoint["image_size"]
        return (int(height), int(width))
    return (64, 512)


def load_model(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    class_to_idx: dict[str, int],
    device: Any,
) -> Any:
    import torch

    if args.model_kind == "e1cnn":
        from med_ocr_research.models.E1Cnn import E1Cnn

        model = E1Cnn(num_classes=len(class_to_idx)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    if args.model_kind == "e2cnn-bilstm":
        from med_ocr_research.models.E2CnnBiLSTM import E2CnnBiLSTM

        model = E2CnnBiLSTM(
            num_classes=len(class_to_idx),
            blank_idx=int(checkpoint.get("blank_idx", 0)),
            default_return_sequence=is_ctc_checkpoint(checkpoint),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    if args.model_kind == "torchscript-classifier":
        return torch.jit.load(args.checkpoint, map_location=device)

    raise ValueError(f"model-kind nao suportado: {args.model_kind}")


def load_predictions_csv(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_csv(args.predictions_csv)
    if args.limit is not None:
        rows = rows.head(args.limit).copy()

    true_column = "text_true" if "text_true" in rows.columns else args.label_column
    pred_column = "text_pred" if "text_pred" in rows.columns else args.prediction_column
    required = {true_column, pred_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV de predicoes: {sorted(missing)}")

    normalized = pd.DataFrame(
        {
            "sample_id": rows[args.sample_id_column] if args.sample_id_column in rows.columns else rows.index,
            "dataset": rows[args.dataset_column] if args.dataset_column in rows.columns else "",
            "split": rows[args.split_column] if args.split_column in rows.columns else "",
            "image_path": rows[args.image_column] if args.image_column in rows.columns else "",
            "text_true": rows[true_column].astype(str),
            "text_pred": rows[pred_column].astype(str),
        }
    )
    if "confidence" in rows.columns:
        normalized["confidence"] = pd.to_numeric(rows["confidence"], errors="coerce")
    return normalized


def enrich_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.copy()
    rows["text_true"] = rows["text_true"].fillna("").astype(str)
    rows["text_pred"] = rows["text_pred"].fillna("").astype(str)
    rows["exact_match"] = rows["text_true"] == rows["text_pred"]
    rows["char_errors"] = [
        levenshtein_distance(reference, hypothesis)
        for reference, hypothesis in zip(rows["text_true"], rows["text_pred"])
    ]
    rows["ref_chars"] = rows["text_true"].str.len()
    rows["cer_sample"] = rows.apply(
        lambda row: row["char_errors"] / row["ref_chars"] if row["ref_chars"] else float(row["char_errors"] > 0),
        axis=1,
    )
    rows["word_errors"] = [
        sequence_edit_distance(reference.split(), hypothesis.split())
        for reference, hypothesis in zip(rows["text_true"], rows["text_pred"])
    ]
    rows["ref_words"] = rows["text_true"].str.split().map(len)
    rows["wer_sample"] = rows.apply(
        lambda row: row["word_errors"] / row["ref_words"] if row["ref_words"] else float(row["word_errors"] > 0),
        axis=1,
    )
    return rows


def build_per_label_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    label_rows = []
    for label, group in predictions.groupby("text_true", dropna=False):
        metrics = aggregate_metrics(group["text_true"], group["text_pred"])
        label_rows.append(
            {
                "text_true": label,
                "samples": metrics.samples,
                "exact_match_accuracy": metrics.exact_match_accuracy,
                "cer": metrics.cer,
                "wer": metrics.wer,
                "correct": metrics.correct,
            }
        )
    return pd.DataFrame(label_rows).sort_values(
        by=["samples", "text_true"],
        ascending=[False, True],
    )


def confidence_statistics(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    if "confidence" not in predictions.columns:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    values = [
        float(value)
        for value in predictions["confidence"].dropna().tolist()
        if math.isfinite(float(value))
    ]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def numeric_statistics(values: pd.Series) -> dict[str, float | int | None]:
    clean_values = [
        float(value)
        for value in pd.to_numeric(values, errors="coerce").dropna().tolist()
        if math.isfinite(float(value))
    ]
    if not clean_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(clean_values),
        "mean": statistics.fmean(clean_values),
        "median": statistics.median(clean_values),
        "std": statistics.pstdev(clean_values) if len(clean_values) > 1 else 0.0,
        "min": min(clean_values),
        "max": max(clean_values),
    }


def per_dataset_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if "dataset" not in predictions.columns:
        return []

    rows = []
    for dataset, group in predictions.groupby("dataset", dropna=False):
        metrics = aggregate_metrics(group["text_true"], group["text_pred"])
        rows.append(
            {
                "dataset": str(dataset),
                "samples": metrics.samples,
                "exact_match_accuracy": metrics.exact_match_accuracy,
                "cer": metrics.cer,
                "wer": metrics.wer,
                "correct": metrics.correct,
            }
        )
    return sorted(rows, key=lambda row: row["dataset"])


def top_error_pairs(predictions: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    errors = predictions[~predictions["exact_match"]]
    if errors.empty:
        return []
    pairs = (
        errors.groupby(["text_true", "text_pred"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "text_true", "text_pred"], ascending=[False, True, True])
        .head(limit)
    )
    return pairs.to_dict(orient="records")


def build_summary(
    args: argparse.Namespace,
    run_name: str,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    metrics = aggregate_metrics(predictions["text_true"], predictions["text_pred"])
    errors = predictions[~predictions["exact_match"]]
    return {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.predictions_csv or args.checkpoint),
        "test_csv": str(args.test_csv) if args.checkpoint else None,
        "primary_metric": "exact_match_accuracy",
        "metrics": asdict(metrics),
        "per_dataset_metrics": per_dataset_metrics(predictions),
        "confidence": confidence_statistics(predictions),
        "text_true_length": numeric_statistics(predictions["text_true"].str.len()),
        "text_pred_length": numeric_statistics(predictions["text_pred"].str.len()),
        "char_error_statistics": numeric_statistics(predictions["char_errors"]),
        "word_error_statistics": numeric_statistics(predictions["word_errors"]),
        "dataset_distribution": label_distribution(predictions["dataset"]) if "dataset" in predictions else [],
        "label_distribution_top20": label_distribution(predictions["text_true"], limit=20),
        "top_error_pairs": top_error_pairs(predictions, limit=20),
        "error_examples": errors[
            ["sample_id", "dataset", "text_true", "text_pred", "char_errors", "word_errors"]
        ].head(20).to_dict(orient="records"),
        "arguments": vars(args),
    }


def draw_bar_chart(
    values: dict[str, float],
    output_path: Path,
    title: str,
    lower_is_better: set[str] | None = None,
) -> None:
    lower_is_better = lower_is_better or set()
    width, height = 900, 520
    margin_left, margin_top, margin_bottom = 180, 70, 80
    chart_width = width - margin_left - 60
    chart_height = height - margin_top - margin_bottom
    bar_gap = 28
    bar_height = max(32, (chart_height - bar_gap * (len(values) - 1)) // max(len(values), 1))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill=(25, 25, 25))
    draw.line(
        (margin_left, margin_top, margin_left, margin_top + chart_height),
        fill=(100, 100, 100),
    )

    for index, (label, value) in enumerate(values.items()):
        y = margin_top + index * (bar_height + bar_gap)
        clipped = max(0.0, min(1.0, float(value)))
        color = (39, 125, 161) if label not in lower_is_better else (177, 91, 70)
        draw.text((30, y + 8), label, fill=(25, 25, 25))
        draw.rectangle(
            (margin_left, y, margin_left + chart_width, y + bar_height),
            outline=(210, 210, 210),
            fill=(245, 245, 245),
        )
        draw.rectangle(
            (margin_left, y, margin_left + int(chart_width * clipped), y + bar_height),
            fill=color,
        )
        draw.text(
            (margin_left + chart_width + 14, y + 8),
            f"{value:.4f}",
            fill=(25, 25, 25),
        )

    image.save(output_path)


def draw_error_histogram(predictions: pd.DataFrame, output_path: Path) -> None:
    bins = {
        "0": int((predictions["char_errors"] == 0).sum()),
        "1": int((predictions["char_errors"] == 1).sum()),
        "2": int((predictions["char_errors"] == 2).sum()),
        "3": int((predictions["char_errors"] == 3).sum()),
        "4+": int((predictions["char_errors"] >= 4).sum()),
    }
    total = max(sum(bins.values()), 1)
    draw_bar_chart(
        {label: count / total for label, count in bins.items()},
        output_path,
        "Distribuicao de erros por caracteres",
    )


def write_markdown_report(
    summary: dict[str, Any],
    per_label_path: Path,
    predictions_path: Path,
    chart_paths: list[Path],
    output_path: Path,
) -> None:
    metrics = summary["metrics"]
    confidence = summary["confidence"]
    lines = [
        f"# Evaluation - {summary['run_name']}",
        "",
        f"- Fonte: `{summary['source']}`",
        f"- Amostras: {metrics['samples']}",
        f"- Metrica primaria: `exact_match_accuracy`",
        f"- Palavras Exatas: {metrics['exact_match_accuracy']:.6f}",
        f"- CER: {metrics['cer']:.6f}",
        f"- WER: {metrics['wer']:.6f}",
        f"- Corretas: {metrics['correct']}",
        f"- Erros: {metrics['samples'] - metrics['correct']}",
        "",
        "## Confianca",
        "",
        f"- Predicoes com confianca: {confidence['count']}",
        f"- Media: {format_optional_float(confidence['mean'])}",
        f"- Mediana: {format_optional_float(confidence['median'])}",
        f"- Desvio padrao: {format_optional_float(confidence['std'])}",
        f"- Minimo: {format_optional_float(confidence['min'])}",
        f"- Maximo: {format_optional_float(confidence['max'])}",
        "",
        "## Artefatos",
        "",
        f"- Predicoes avaliadas: `{predictions_path}`",
        f"- Metricas por palavra/classe: `{per_label_path}`",
    ]
    lines.extend(f"- Grafico: `{chart_path}`" for chart_path in chart_paths)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    run_name = infer_run_name(args)
    run_output_dir = args.output_dir / run_name
    predictions_output_dir = args.predictions_output_dir or run_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_output_dir.mkdir(parents=True, exist_ok=True)

    predictions = (
        load_predictions_csv(args)
        if args.predictions_csv is not None
        else generate_predictions_from_checkpoint(args)
    )
    predictions = enrich_predictions(predictions)
    per_label = build_per_label_metrics(predictions)
    summary = build_summary(args, run_name, predictions)

    predictions_path = predictions_output_dir / "evaluated_predictions.csv"
    metrics_path = run_output_dir / "metrics.json"
    per_label_path = run_output_dir / "per_label_metrics.csv"
    metrics_chart_path = run_output_dir / "metrics_chart.png"
    error_chart_path = run_output_dir / "char_error_distribution.png"
    report_path = run_output_dir / "evaluation.md"

    predictions.to_csv(predictions_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    metrics_path.write_text(
        json.dumps(make_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_bar_chart(
        {
            "Palavras Exatas": summary["metrics"]["exact_match_accuracy"],
            "CER": summary["metrics"]["cer"],
            "WER": summary["metrics"]["wer"],
        },
        metrics_chart_path,
        "Metricas principais da run",
        lower_is_better={"CER", "WER"},
    )
    draw_error_histogram(predictions, error_chart_path)
    write_markdown_report(
        summary=summary,
        per_label_path=per_label_path,
        predictions_path=predictions_path,
        chart_paths=[metrics_chart_path, error_chart_path],
        output_path=report_path,
    )

    print(f"Run: {run_name}")
    print(f"Palavras Exatas: {summary['metrics']['exact_match_accuracy']:.6f}")
    print(f"CER: {summary['metrics']['cer']:.6f}")
    print(f"WER: {summary['metrics']['wer']:.6f}")
    print(f"Metricas: {metrics_path}")
    print(f"Predicoes avaliadas: {predictions_path}")
    print(f"Relatorio: {report_path}")


if __name__ == "__main__":
    main()
