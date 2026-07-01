from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


MANIFEST_COLUMNS = [
    "sample_id",
    "dataset",
    "image_path",
    "text_raw",
    "text_normalized",
    "split",
    "writer_id",
    "document_id",
    "width",
    "height",
    "notes",
    "source_order",
    "source_label_file",
    "source_image_path",
    "generic_name",
    "original_width",
    "original_height",
]


@dataclass(frozen=True)
class DatasetSource:
    dataset: str
    prefix: str
    label_file: Path
    image_root: Path
    image_col: str
    text_col: str
    split: str
    generic_col: str | None = None


@dataclass(frozen=True)
class ProcessedSample:
    sample_id: str
    dataset: str
    image_path: str
    text_raw: str
    text_normalized: str
    split: str
    writer_id: str
    document_id: str
    width: int
    height: int
    notes: str
    source_order: int
    source_label_file: str
    source_image_path: str
    generic_name: str
    original_width: int
    original_height: int


def project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]


def path_for_csv(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def resize_to_512_square(
    source_path: Path,
    target_path: Path,
    size: int,
    padding_value: int = 255,
) -> tuple[int, int]:
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        original_width, original_height = image.size
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (size, size), (padding_value,) * 3)
        left = (size - image.width) // 2
        top = (size - image.height) // 2
        canvas.paste(image, (left, top))
        canvas.save(target_path, format="PNG")

    return original_width, original_height


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def find_doctor_dataset_root(raw_root: Path) -> Path:
    candidates = [
        path
        for path in raw_root.iterdir()
        if path.is_dir()
        and path.name.lower().startswith("doctor")
        and "prescription" in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Doctor's Handwritten Prescription BD dataset folder was not found "
            f"inside {raw_root}."
        )
    return candidates[0]


def discover_sources(raw_root: Path) -> list[DatasetSource]:
    rx_root = raw_root / "RxHandBD-Raw"
    doctor_root = find_doctor_dataset_root(raw_root)

    return [
        DatasetSource(
            dataset="RxHandBD",
            prefix="rxhandbd",
            label_file=rx_root / "Prescription_Labels.csv",
            image_root=rx_root / "Images",
            image_col="Images",
            text_col="Text",
            split="",
        ),
        DatasetSource(
            dataset="PrescriptionBD",
            prefix="prescriptionbd_train",
            label_file=doctor_root / "Training" / "training_labels.csv",
            image_root=doctor_root / "Training" / "training_words",
            image_col="IMAGE",
            text_col="MEDICINE_NAME",
            generic_col="GENERIC_NAME",
            split="train",
        ),
        DatasetSource(
            dataset="PrescriptionBD",
            prefix="prescriptionbd_val",
            label_file=doctor_root / "Validation" / "validation_labels.csv",
            image_root=doctor_root / "Validation" / "validation_words",
            image_col="IMAGE",
            text_col="MEDICINE_NAME",
            generic_col="GENERIC_NAME",
            split="val",
        ),
        DatasetSource(
            dataset="PrescriptionBD",
            prefix="prescriptionbd_test",
            label_file=doctor_root / "Testing" / "testing_labels.csv",
            image_root=doctor_root / "Testing" / "testing_words",
            image_col="IMAGE",
            text_col="MEDICINE_NAME",
            generic_col="GENERIC_NAME",
            split="test",
        ),
    ]


def validate_source(source: DatasetSource) -> None:
    if not source.label_file.exists():
        raise FileNotFoundError(f"Label file not found: {source.label_file}")
    if not source.image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {source.image_root}")


def process_source(
    source: DatasetSource,
    data_root: Path,
    output_image_dir: Path,
    size: int,
) -> Iterable[ProcessedSample]:
    validate_source(source)
    rows = read_rows(source.label_file)

    for source_order, row in enumerate(rows, start=1):
        image_name = (row.get(source.image_col) or "").strip()
        text_raw = (row.get(source.text_col) or "").strip()
        generic_name = (
            (row.get(source.generic_col) or "").strip()
            if source.generic_col is not None
            else ""
        )

        if not image_name:
            raise ValueError(
                f"Missing image name in {source.label_file} at row {source_order}."
            )

        source_image = source.image_root / image_name
        if not source_image.exists():
            raise FileNotFoundError(
                f"Image referenced by CSV was not found: {source_image}"
            )

        sample_id = f"{source.prefix}_{source_order:06d}"
        output_image = output_image_dir / f"{sample_id}.png"
        original_width, original_height = resize_to_512_square(
            source_image,
            output_image,
            size,
        )

        notes = "resized_with_aspect_ratio_and_white_padding"
        if not text_raw:
            notes += ";empty_text"

        yield ProcessedSample(
            sample_id=sample_id,
            dataset=source.dataset,
            image_path=path_for_csv(output_image, data_root),
            text_raw=text_raw,
            text_normalized=normalize_text(text_raw),
            split=source.split,
            writer_id="",
            document_id="",
            width=size,
            height=size,
            notes=notes,
            source_order=source_order,
            source_label_file=path_for_csv(source.label_file, data_root),
            source_image_path=path_for_csv(source_image, data_root),
            generic_name=generic_name,
            original_width=original_width,
            original_height=original_height,
        )


def write_manifest(samples: list[ProcessedSample], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def process_all(project_root: Path, image_size: int) -> Path:
    data_root = project_root / "data"
    raw_root = data_root / "raw"
    processed_root = data_root / "processed"
    output_image_dir = processed_root / "WritingProcessed"
    output_csv = processed_root / "WritingProcessed.csv"

    output_image_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[ProcessedSample] = []
    for source in discover_sources(raw_root):
        print(f"Processing {source.dataset} ({source.split or 'no split'})...")
        source_samples = list(
            process_source(
                source=source,
                data_root=data_root,
                output_image_dir=output_image_dir,
                size=image_size,
            )
        )
        all_samples.extend(source_samples)
        print(f"  {len(source_samples)} samples processed")

    write_manifest(all_samples, output_csv)
    print(f"Unified manifest written to: {output_csv}")
    print(f"Resized images written to: {output_image_dir}")
    print(f"Total samples: {len(all_samples)}")
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize raw dataset images to 512x512 and create a unified CSV "
            "manifest while preserving source CSV order."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root_from_this_file(),
        help="Path to the SourceCode project root.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Final square image size in pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_all(project_root=args.project_root.resolve(), image_size=args.image_size)


if __name__ == "__main__":
    main()
