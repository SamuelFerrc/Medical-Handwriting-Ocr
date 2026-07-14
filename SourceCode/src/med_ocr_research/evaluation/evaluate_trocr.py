from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.evaluation.evaluate_model import (  # noqa: E402
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


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/runs/e3.03/trocr_best"
DEFAULT_SCENARIOS_DIR = PROJECT_ROOT / "data/splits/scenarios"
DEFAULT_TEST_CSV = DEFAULT_SCENARIOS_DIR / "combined/test.csv"
DEFAULT_IMAGE_SIZE = (384, 384)
DEFAULT_MAX_GENERATION_LENGTH = 128
DEFAULT_NUM_BEAMS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia um checkpoint TrOCR usando exatamente os mesmos artefatos "
            "gerados para E1CNN/E2CNN."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help=(
            "CSV de teste. Se omitido, tenta usar o dataset salvo em "
            "training_metadata.json e cai para o teste combined."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
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
        return sanitize_name(f"{metadata_run_name}_trocr_evaluation")

    checkpoint_name = args.checkpoint.name
    if checkpoint_name in {"trocr_best", "trocr_final"}:
        checkpoint_name = args.checkpoint.parent.name
    return sanitize_name(f"{checkpoint_name}_trocr_evaluation")


def resolve_image_size(
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> tuple[int, int]:
    metadata_image_size = metadata.get("image_size") or DEFAULT_IMAGE_SIZE
    image_height = args.image_height or int(metadata_image_size[0])
    image_width = args.image_width or int(metadata_image_size[1])
    return image_height, image_width


def resolve_generation_config(
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> tuple[int, int]:
    max_generation_length = args.max_generation_length or int(
        metadata.get("max_generation_length", DEFAULT_MAX_GENERATION_LENGTH)
    )
    num_beams = args.num_beams or int(metadata.get("num_beams", DEFAULT_NUM_BEAMS))
    return max_generation_length, num_beams


def prepare_args(args: argparse.Namespace, metadata: dict[str, Any]) -> argparse.Namespace:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint TrOCR ausente: {args.checkpoint}")

    args.predictions_csv = None
    args.test_csv = infer_test_csv(args, metadata)
    if not args.test_csv.exists():
        raise FileNotFoundError(f"CSV de teste ausente: {args.test_csv}")

    args.label_column = args.label_column or metadata.get("label_column", "text_normalized")
    args.image_column = args.image_column or metadata.get("image_column", "image_path")
    args.image_height, args.image_width = resolve_image_size(args, metadata)
    args.max_generation_length, args.num_beams = resolve_generation_config(args, metadata)
    return args


def load_image(image_path: Path, image_size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image.resize((image_size[1], image_size[0]))


def load_trocr(checkpoint_dir: Path) -> tuple[Any, Any]:
    try:
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers nao esta instalado. Rode: "
            "python -m pip install -r requirements-train.txt"
        ) from exc

    try:
        processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        try:
            processor = TrOCRProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
        except TypeError:
            processor = TrOCRProcessor(
                feature_extractor=image_processor,
                tokenizer=tokenizer,
            )

    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    return processor, model


def generation_confidences(model: Any, generated: Any, count: int) -> list[float]:
    scores = getattr(generated, "scores", None)
    sequences = getattr(generated, "sequences", None)
    if not scores or sequences is None:
        return [float("nan")] * count

    try:
        import torch

        beam_indices = getattr(generated, "beam_indices", None)
        transition_scores = model.compute_transition_scores(
            sequences,
            scores,
            beam_indices,
            normalize_logits=True,
        )
        confidences = []
        for sequence_scores in transition_scores:
            valid_scores = sequence_scores[torch.isfinite(sequence_scores)]
            if valid_scores.numel() == 0:
                confidences.append(float("nan"))
            else:
                confidences.append(float(torch.exp(valid_scores).mean().item()))
    except Exception:
        confidences = fallback_generation_confidences(scores, count)

    if len(confidences) < count:
        confidences.extend([float("nan")] * (count - len(confidences)))
    return confidences[:count]


def fallback_generation_confidences(scores: tuple[Any, ...], count: int) -> list[float]:
    import torch

    confidences = []
    for batch_index in range(count):
        token_confidences = []
        for step_scores in scores:
            if batch_index >= step_scores.size(0):
                continue
            probabilities = torch.softmax(step_scores[batch_index], dim=-1)
            token_confidences.append(float(probabilities.max().item()))
        if token_confidences:
            confidences.append(sum(token_confidences) / len(token_confidences))
        else:
            confidences.append(float("nan"))
    return confidences


def load_test_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_csv(args.test_csv)
    if args.split_column in rows.columns:
        rows = rows[rows[args.split_column].astype(str).str.lower() == "test"].copy()

    required = {args.image_column, args.label_column}
    missing = required.difference(rows.columns)
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


def batched_rows(rows: pd.DataFrame, batch_size: int) -> list[pd.DataFrame]:
    safe_batch_size = max(1, batch_size)
    return [
        rows.iloc[start : start + safe_batch_size]
        for start in range(0, len(rows), safe_batch_size)
    ]


def generate_predictions(args: argparse.Namespace) -> pd.DataFrame:
    import torch

    device = choose_device(args.device)
    image_size = (args.image_height, args.image_width)
    rows = load_test_rows(args)
    processor, model = load_trocr(args.checkpoint)
    model.to(device)
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(batched_rows(rows, args.batch_size), desc="Avaliacao", leave=False):
            images = [
                load_image(
                    resolve_image_path(args.data_root, row[args.image_column]),
                    image_size,
                )
                for _, row in batch.iterrows()
            ]
            pixel_values = processor(
                images=images,
                return_tensors="pt",
            ).pixel_values.to(device)
            generated = model.generate(
                pixel_values,
                max_length=args.max_generation_length,
                num_beams=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
            batch_predictions = processor.batch_decode(
                generated.sequences,
                skip_special_tokens=True,
            )
            confidences = generation_confidences(
                model=model,
                generated=generated,
                count=len(batch_predictions),
            )

            for offset, (_, row) in enumerate(batch.iterrows()):
                confidence = confidences[offset]
                prediction_rows.append(
                    {
                        "sample_id": row.get(args.sample_id_column, len(prediction_rows)),
                        "dataset": row.get(args.dataset_column, ""),
                        "split": row.get(args.split_column, "test"),
                        "image_path": row.get(args.image_column, ""),
                        "text_true": str(row[args.label_column]),
                        "text_pred": str(batch_predictions[offset]),
                        "confidence": None if math.isnan(confidence) else confidence,
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
