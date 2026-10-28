from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/runs/e3.03_seed42/trocr_best"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data/splits/scenarios/prescriptionbd/test.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_IMAGE_SIZE = (384, 384)
DEFAULT_MAX_GENERATION_LENGTH = 128
DEFAULT_NUM_BEAMS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa o TrOCR treinado e imprime palavra real e predicao."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--max-generation-length", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    return parser.parse_args()


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    return data_root / path


def load_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "training_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_image_size(args: argparse.Namespace, metadata: dict[str, Any]) -> tuple[int, int]:
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


def load_image(image_path: Path, image_size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image.resize((image_size[1], image_size[0]))


def load_trocr(checkpoint_dir: Path):
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


def generation_confidence(scores: tuple[torch.Tensor, ...] | None) -> float:
    if not scores:
        return float("nan")

    token_confidences = []
    for step_scores in scores:
        probabilities = torch.softmax(step_scores[0], dim=-1)
        token_confidences.append(float(probabilities.max().item()))
    return sum(token_confidences) / len(token_confidences)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA disponivel: usando GPU {torch.cuda.get_device_name(0)}")

    metadata = load_metadata(args.checkpoint)
    image_size = resolve_image_size(args, metadata)
    max_generation_length, num_beams = resolve_generation_config(args, metadata)
    processor, model = load_trocr(args.checkpoint)
    model.to(device)
    model.eval()

    rows = pd.read_csv(args.test_csv)
    if "split" in rows.columns:
        rows = rows[rows["split"].astype(str).str.lower() == "test"].copy()
    rows = rows.sample(
        n=min(args.num_samples, len(rows)),
        random_state=args.seed,
    ).reset_index(drop=True)

    print(f"Dispositivo: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"CSV de teste: {args.test_csv}")
    print(f"Tamanho da imagem: {image_size[0]}x{image_size[1]}")
    print(f"Max generation length: {max_generation_length}")
    print(f"Num beams: {num_beams}")
    print()

    correct = 0
    with torch.no_grad():
        for index, row in rows.iterrows():
            image_path = resolve_image_path(args.data_root, row[args.image_column])
            image = load_image(image_path, image_size)
            pixel_values = processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.to(device)
            generated = model.generate(
                pixel_values,
                max_length=max_generation_length,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
            predicted_word = processor.batch_decode(
                generated.sequences,
                skip_special_tokens=True,
            )[0]
            confidence = generation_confidence(generated.scores)
            real_word = str(row[args.label_column])
            is_correct = predicted_word == real_word
            correct += int(is_correct)

            print(f"Amostra {index + 1}")
            print(f"Imagem: {image_path}")
            print(f"Palavra real: {real_word}")
            print(
                f"Predicao: {predicted_word} "
                f"(confianca: {confidence:.4f})"
            )
            print(f"Acertou: {'sim' if is_correct else 'nao'}")
            print()

    accuracy = correct / len(rows) if len(rows) else 0.0
    print(f"Acuracia nas {len(rows)} amostras impressas: {accuracy:.4f}")


if __name__ == "__main__":
    main()
