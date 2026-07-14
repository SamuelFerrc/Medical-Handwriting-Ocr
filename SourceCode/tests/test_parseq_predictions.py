from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.evaluation.evaluate_model import resolve_image_path
from med_ocr_research.evaluation.evaluate_parseq import load_image_tensor
from med_ocr_research.models.parseq_utils import (
    decode_parseq_logits,
    load_parseq_bundle,
)


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/runs/e4.03/parseq_best"
DEFAULT_TEST_CSV = PROJECT_ROOT / "data/splits/scenarios/combined/test.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa o PARSeq treinado e imprime texto real e predicao."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "training_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(args.checkpoint)
    image_size = tuple(int(value) for value in metadata["image_size"])
    max_generation_length = int(metadata["max_generation_length"])
    model, _ = load_parseq_bundle(args.checkpoint, map_location="cpu")
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
    print()

    correct = 0
    with torch.no_grad():
        for index, row in rows.iterrows():
            image_path = resolve_image_path(args.data_root, row[args.image_column])
            image = load_image_tensor(image_path, image_size).unsqueeze(0).to(device)
            logits = model(image, max_length=max_generation_length)
            predictions, confidences = decode_parseq_logits(model, logits)
            predicted_text = predictions[0]
            confidence = confidences[0]
            real_text = str(row[args.label_column])
            is_correct = predicted_text == real_text
            correct += int(is_correct)

            print(f"Amostra {index + 1}")
            print(f"Imagem: {image_path}")
            print(f"Texto real: {real_text}")
            print(f"Predicao: {predicted_text} (confianca: {confidence:.4f})")
            print(f"Acertou: {'sim' if is_correct else 'nao'}")
            print()

    accuracy = correct / len(rows) if len(rows) else 0.0
    print(f"Acuracia nas {len(rows)} amostras impressas: {accuracy:.4f}")


if __name__ == "__main__":
    main()
