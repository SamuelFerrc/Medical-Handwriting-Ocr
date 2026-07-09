from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from med_ocr_research.models.E2CnnBiLSTM import E2CnnBiLSTM


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/runs/e2.02/e2cnn_bilstm_best_checkpoint.pt"
)
DEFAULT_TEST_CSV = PROJECT_ROOT / "data/splits/scenarios/combined/test.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa a E2CnnBiLSTM treinada e imprime palavra real e predicao."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-column", default="text_normalized")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    return data_root / path


def load_image(image_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    with Image.open(image_path) as image:
        if image.mode == "RGB":
            image = image.convert("L")
        elif image.mode != "L":
            image = image.convert("L")
        image = image.resize((image_size[1], image_size[0]))
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    return torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA disponivel: usando GPU {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    uses_ctc = checkpoint.get("target_kind") == "ctc"
    if uses_ctc and "char_to_idx" in checkpoint:
        class_to_idx = checkpoint["char_to_idx"]
    else:
        class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    image_size = tuple(checkpoint.get("image_size", (64, 512)))

    model = E2CnnBiLSTM(
        num_classes=len(class_to_idx),
        blank_idx=int(checkpoint.get("blank_idx", 0)),
        default_return_sequence=uses_ctc,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    print()

    correct = 0
    with torch.no_grad():
        for index, row in rows.iterrows():
            image_path = resolve_image_path(args.data_root, row[args.image_column])
            image_tensor = load_image(image_path, image_size).to(device)
            logits = model(image_tensor)
            if uses_ctc:
                predicted_word = model.ctc_decode(
                    logits,
                    idx_to_token=idx_to_class,
                    blank_idx=int(checkpoint.get("blank_idx", 0)),
                )[0]
                probabilities = torch.softmax(logits, dim=-1)
                confidence = probabilities.max(dim=-1).values.mean()
            else:
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_idx = probabilities.max(dim=1)
                predicted_word = idx_to_class[int(predicted_idx.item())]
            real_word = str(row[args.label_column])
            is_correct = predicted_word == real_word
            correct += int(is_correct)

            print(f"Amostra {index + 1}")
            print(f"Imagem: {image_path}")
            print(f"Palavra real: {real_word}")
            print(
                f"Predicao: {predicted_word} "
                f"(confianca: {confidence.item():.4f})"
            )
            print(f"Acertou: {'sim' if is_correct else 'nao'}")
            print()

    accuracy = correct / len(rows) if len(rows) else 0.0
    print(f"Acuracia nas {len(rows)} amostras impressas: {accuracy:.4f}")


if __name__ == "__main__":
    main()
