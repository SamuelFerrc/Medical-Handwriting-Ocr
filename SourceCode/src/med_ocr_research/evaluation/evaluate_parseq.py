from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.evaluation.evaluate_model import (
    DEFAULT_DATA_ROOT,
    DEFAULT_METRICS_DIR,
    build_per_label_metrics,
    build_summary,
    choose_device,
    draw_bar_chart,
    draw_error_histogram,
    enrich_predictions,
    make_json_safe,
    resolve_image_path,
    sanitize_name,
    write_markdown_report,
)
from med_ocr_research.models.parseq_utils import (
    decode_parseq_logits,
    load_parseq_bundle,
    parseq_input_size,
    parseq_max_label_length,
    validate_parseq_labels,
)
from med_ocr_research.training.train_parseq import build_image_transform


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/runs/e4.03/parseq_best"
DEFAULT_SCENARIOS_DIR = PROJECT_ROOT / "data/splits/scenarios"
DEFAULT_TEST_CSV = DEFAULT_SCENARIOS_DIR / "combined/test.csv"
DEFAULT_IMAGE_SIZE = (64, 512)
DEFAULT_MAX_GENERATION_LENGTH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia um checkpoint PARSeq usando os mesmos artefatos e metricas "
            "do TrOCR."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--predictions-output-dir", type=Path, default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--dataset-column", default="dataset")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--max-generation-length", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "training_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def infer_test_csv(args: argparse.Namespace, metadata: dict[str, Any]) -> Path:
    if args.test_csv is not None:
        return args.test_csv
    dataset = metadata.get("dataset")
    if dataset:
        candidate = DEFAULT_SCENARIOS_DIR / str(dataset) / "test.csv"
        if candidate.exists():
            return candidate
    return DEFAULT_TEST_CSV


def infer_run_name(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    if args.run_name:
        return sanitize_name(args.run_name)
    metadata_run_name = metadata.get("run_name")
    if metadata_run_name:
        return sanitize_name(f"{metadata_run_name}_parseq_evaluation")
    checkpoint_name = args.checkpoint.name
    if checkpoint_name in {"parseq_best", "parseq_final"}:
        checkpoint_name = args.checkpoint.parent.name
    return sanitize_name(f"{checkpoint_name}_parseq_evaluation")


def prepare_args(
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> argparse.Namespace:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint PARSeq ausente: {args.checkpoint}")
    args.predictions_csv = None
    args.test_csv = infer_test_csv(args, metadata)
    if not args.test_csv.exists():
        raise FileNotFoundError(f"CSV de teste ausente: {args.test_csv}")
    args.label_column = args.label_column or metadata.get(
        "label_column", "text_normalized"
    )
    args.image_column = args.image_column or metadata.get("image_column", "image_path")
    metadata_size = metadata.get("image_size") or DEFAULT_IMAGE_SIZE
    args.image_height = args.image_height or int(metadata_size[0])
    args.image_width = args.image_width or int(metadata_size[1])
    args.max_generation_length = args.max_generation_length or int(
        metadata.get("max_generation_length", DEFAULT_MAX_GENERATION_LENGTH)
    )
    args.num_beams = args.num_beams or int(metadata.get("num_beams", 1))
    if args.num_beams != 1:
        raise ValueError("PARSeq suporta apenas --num-beams 1 neste pipeline.")
    if args.batch_size < 1:
        raise ValueError("--batch-size precisa ser maior que zero.")
    return args


def load_test_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_csv(args.test_csv)
    if args.split_column in rows.columns:
        rows = rows[rows[args.split_column].astype(str).str.lower() == "test"].copy()
    missing = {args.image_column, args.label_column}.difference(rows.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV de teste: {sorted(missing)}")
    rows = rows.dropna(subset=[args.image_column, args.label_column]).copy()
    rows[args.label_column] = rows[args.label_column].astype(str)
    rows = rows[rows[args.label_column].str.len() > 0].copy()
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    if rows.empty:
        raise ValueError(f"Nenhuma amostra encontrada em {args.test_csv}")
    return rows.reset_index(drop=True)


def load_image_tensor(image_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size[1], image_size[0]))
        return build_image_transform()(image)


def generate_predictions(args: argparse.Namespace) -> pd.DataFrame:
    device = choose_device(args.device)
    rows = load_test_rows(args)
    model, _ = load_parseq_bundle(args.checkpoint, map_location="cpu")
    expected_size = parseq_input_size(model)
    requested_size = (args.image_height, args.image_width)
    if requested_size != expected_size:
        raise ValueError(
            f"O checkpoint requer imagens {expected_size}, mas foi solicitado "
            f"{requested_size}."
        )
    if args.max_generation_length > parseq_max_label_length(model):
        raise ValueError("Max generation length excede a capacidade do checkpoint.")
    validate_parseq_labels(
        rows[args.label_column].tolist(),
        model,
        parseq_max_label_length(model),
        "test",
    )
    model.to(device)
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(rows), args.batch_size),
            desc="Avaliacao",
            leave=False,
        ):
            batch = rows.iloc[start : start + args.batch_size]
            images = torch.stack(
                [
                    load_image_tensor(
                        resolve_image_path(args.data_root, row[args.image_column]),
                        requested_size,
                    )
                    for _, row in batch.iterrows()
                ]
            ).to(device)
            logits = model(images, max_length=args.max_generation_length)
            predictions, confidences = decode_parseq_logits(model, logits)
            for offset, (_, row) in enumerate(batch.iterrows()):
                prediction_rows.append(
                    {
                        "sample_id": row.get(
                            args.sample_id_column, len(prediction_rows)
                        ),
                        "dataset": row.get(args.dataset_column, ""),
                        "split": row.get(args.split_column, "test"),
                        "image_path": row.get(args.image_column, ""),
                        "text_true": str(row[args.label_column]),
                        "text_pred": predictions[offset],
                        "confidence": confidences[offset],
                    }
                )
    return pd.DataFrame(prediction_rows)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.checkpoint)
    args = prepare_args(args, metadata)
    run_name = infer_run_name(args, metadata)
    run_output_dir = args.output_dir / run_name
    predictions_output_dir = args.predictions_output_dir or run_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_output_dir.mkdir(parents=True, exist_ok=True)

    predictions = enrich_predictions(generate_predictions(args))
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
        summary,
        per_label_path,
        predictions_path,
        [metrics_chart_path, error_chart_path],
        report_path,
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
