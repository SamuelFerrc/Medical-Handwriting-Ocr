from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
GROUP_COLUMNS = ("writer_id", "document_id")
SCENARIOS = {
    "rxhandbd": {
        "label": "RxHandBD",
        "datasets": ("RxHandBD",),
    },
    "prescriptionbd": {
        "label": "PrescriptionBD",
        "datasets": ("PrescriptionBD",),
    },
    "combined": {
        "label": "Combinado",
        "datasets": ("RxHandBD", "PrescriptionBD"),
    },
}


def project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    required = {
        "sample_id",
        "dataset",
        "image_path",
        "text_raw",
        "text_normalized",
        "split",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return df


def split_counts(total: int) -> dict[str, int]:
    train = round(total * SPLIT_RATIOS["train"])
    val = round(total * SPLIT_RATIOS["val"])
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def usable_group_column(df: pd.DataFrame) -> str | None:
    for column in GROUP_COLUMNS:
        if column in df.columns and (df[column].astype(str).str.strip() != "").any():
            return column
    return None


def assign_by_rows(df: pd.DataFrame, seed: int) -> pd.Series:
    indices = list(df.index)
    random.Random(seed).shuffle(indices)
    counts = split_counts(len(indices))
    assigned: dict[int, str] = {}
    cursor = 0
    for split in SPLITS:
        next_cursor = cursor + counts[split]
        for index in indices[cursor:next_cursor]:
            assigned[index] = split
        cursor = next_cursor
    return pd.Series(assigned)


def assign_by_group(df: pd.DataFrame, group_column: str, seed: int) -> pd.Series:
    groups = []
    for group_value, group_df in df.groupby(group_column, sort=False):
        groups.append((str(group_value), list(group_df.index), len(group_df)))

    random.Random(seed).shuffle(groups)
    targets = split_counts(len(df))
    assigned: dict[int, str] = {}
    current_counts = {split: 0 for split in SPLITS}

    for _, indices, size in groups:
        split = min(
            SPLITS,
            key=lambda name: current_counts[name] / max(targets[name], 1),
        )
        for index in indices:
            assigned[index] = split
        current_counts[split] += size

    return pd.Series(assigned)


def create_splits(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    result = df.copy()
    result["split"] = result["split"].astype(str).str.strip().str.lower()
    methods: dict[str, str] = {}

    for dataset, dataset_df in result.groupby("dataset", sort=True):
        existing = set(dataset_df["split"].dropna().astype(str).str.strip())
        has_official_split = existing.issubset(set(SPLITS)) and set(SPLITS).issubset(existing)

        if has_official_split and (dataset_df["split"] != "").all():
            methods[dataset] = "split original preservado (train/val/test ja presente)"
            continue

        group_column = usable_group_column(dataset_df)
        dataset_seed = seed + sum(ord(char) for char in str(dataset))
        if group_column:
            assignments = assign_by_group(dataset_df, group_column, dataset_seed)
            methods[dataset] = (
                f"split 70/15/15 gerado por {group_column}, seed {dataset_seed}"
            )
        else:
            assignments = assign_by_rows(dataset_df, dataset_seed)
            methods[dataset] = f"split 70/15/15 gerado por amostra, seed {dataset_seed}"

        for index, split in assignments.items():
            result.at[index, "split"] = split

    return result, methods


def repair_image_paths(df: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    result = df.copy()
    before_missing = 0
    after_missing = 0
    repaired = 0
    fallback_dir = data_root / "processed" / "WritingProcessed"

    for index, image_path in result["image_path"].astype(str).items():
        current = data_root / image_path
        if current.exists():
            continue

        before_missing += 1
        fallback = fallback_dir / Path(image_path).name
        if fallback.exists():
            result.at[index, "image_path"] = fallback.relative_to(data_root).as_posix()
            repaired += 1
        else:
            after_missing += 1

    return result, {
        "missing_before_repair": before_missing,
        "repaired": repaired,
        "missing_after_repair": after_missing,
    }


def summarize_text_lengths(df: pd.DataFrame) -> dict[str, float | int]:
    chars = df["text_normalized"].astype(str).str.len()
    words = df["text_normalized"].astype(str).str.split().str.len()
    return {
        "chars_mean": round(float(chars.mean()), 2),
        "chars_median": round(float(chars.median()), 2),
        "chars_min": int(chars.min()),
        "chars_max": int(chars.max()),
        "words_mean": round(float(words.mean()), 2),
        "words_median": round(float(words.median()), 2),
        "words_min": int(words.min()),
        "words_max": int(words.max()),
    }


def dataframe_table(records: list[dict[str, object]], columns: list[str]) -> str:
    if not records:
        return ""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in records:
        rows.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def project_relative(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def scenario_key(name: str) -> str:
    normalized = (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )
    if normalized == "rxhandbd":
        return "rxhandbd"
    if normalized in {"prescriptionbd", "doctorshandwrittenprescriptionbd"}:
        return "prescriptionbd"
    if normalized in {"combined", "combinado"}:
        return "combined"
    raise ValueError(f"Unknown dataset scenario: {name}")


def scenario_subset(df: pd.DataFrame, key: str) -> pd.DataFrame:
    datasets = SCENARIOS[key]["datasets"]
    return df[df["dataset"].isin(datasets)].copy()


def write_scenario_files(
    df: pd.DataFrame,
    split_dir: Path,
    project_root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    scenario_root = split_dir / "scenarios"
    records: list[dict[str, object]] = []
    lookup: dict[str, dict[str, str]] = {}

    for key, definition in SCENARIOS.items():
        scenario_dir = scenario_root / key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        subset = scenario_subset(df, key)
        files: dict[str, str] = {}

        all_path = scenario_dir / "all.csv"
        subset.to_csv(all_path, index=False)
        files["all"] = project_relative(all_path, project_root)
        records.append(
            {
                "scenario": definition["label"],
                "split": "all",
                "samples": len(subset),
                "path": files["all"],
            }
        )

        for split in SPLITS:
            split_subset = subset[subset["split"] == split]
            split_path = scenario_dir / f"{split}.csv"
            split_subset.to_csv(split_path, index=False)
            files[split] = project_relative(split_path, project_root)

            alias_path = split_dir / f"{split}_{key}.csv"
            split_subset.to_csv(alias_path, index=False)

            records.append(
                {
                    "scenario": definition["label"],
                    "split": split,
                    "samples": len(split_subset),
                    "path": files[split],
                }
            )

        lookup[key] = files

    pd.DataFrame(records).to_csv(split_dir / "scenario_datasets.csv", index=False)
    return records, lookup


def read_experiment_matrix(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return list(data.get("experiments", []))


def write_experiment_dataset_map(
    experiments: list[dict[str, object]],
    scenario_lookup: dict[str, dict[str, str]],
    split_dir: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for experiment in experiments:
        train_key = scenario_key(str(experiment["train_dataset"]))
        test_key = scenario_key(str(experiment["test_dataset"]))
        records.append(
            {
                "id": experiment["id"],
                "model": experiment["model"],
                "scenario": experiment["scenario"],
                "train_dataset": SCENARIOS[train_key]["label"],
                "test_dataset": SCENARIOS[test_key]["label"],
                "train_manifest": scenario_lookup[train_key]["train"],
                "val_manifest": scenario_lookup[train_key]["val"],
                "test_manifest": scenario_lookup[test_key]["test"],
            }
        )

    if records:
        pd.DataFrame(records).to_csv(
            split_dir / "experiment_datasets.csv",
            index=False,
        )
        (split_dir / "experiment_datasets.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return records


def build_statistics(
    df: pd.DataFrame,
    methods: dict[str, str],
    image_path_stats: dict[str, int],
    seed: int,
    scenario_records: list[dict[str, object]],
    experiment_records: list[dict[str, object]],
) -> dict[str, object]:
    by_dataset_split = (
        df.groupby(["dataset", "split"], sort=True)
        .size()
        .reset_index(name="samples")
        .to_dict(orient="records")
    )
    by_split = df.groupby("split", sort=True).size().reset_index(name="samples")
    by_dataset = df.groupby("dataset", sort=True).size().reset_index(name="samples")

    text_by_dataset = []
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        row = {"dataset": dataset}
        row.update(summarize_text_lengths(dataset_df))
        text_by_dataset.append(row)

    resolution_counts = (
        df.assign(resolution=df["width"].astype(str) + "x" + df["height"].astype(str))
        .groupby("resolution", sort=True)
        .size()
        .reset_index(name="samples")
        .to_dict(orient="records")
    )

    characters = Counter("".join(df["text_normalized"].astype(str)))
    top_characters = [
        {"character": character if character != " " else "<space>", "count": count}
        for character, count in characters.most_common(30)
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "total_samples": int(len(df)),
        "split_methods": methods,
        "samples_by_dataset": by_dataset.to_dict(orient="records"),
        "samples_by_split": by_split.to_dict(orient="records"),
        "samples_by_dataset_split": by_dataset_split,
        "scenario_datasets": scenario_records,
        "experiment_datasets": experiment_records,
        "text_lengths_by_dataset": text_by_dataset,
        "processed_resolution_counts": resolution_counts,
        "top_characters": top_characters,
        "quality": {
            "duplicated_sample_id": int(df["sample_id"].duplicated().sum()),
            "empty_text_raw": int((df["text_raw"].astype(str).str.strip() == "").sum()),
            "empty_text_normalized": int(
                (df["text_normalized"].astype(str).str.strip() == "").sum()
            ),
            **image_path_stats,
        },
    }


def write_split_files(df: pd.DataFrame, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    assignment_columns = ["sample_id", "dataset", "split"]
    df[assignment_columns].to_csv(split_dir / "split_assignments.csv", index=False)
    for split in SPLITS:
        df[df["split"] == split].to_csv(split_dir / f"{split}.csv", index=False)


def write_statistics_json(stats: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_markdown_report(
    stats: dict[str, object],
    split_manifest: Path,
    split_dir: Path,
    stats_json: Path,
) -> str:
    methods = [
        {"dataset": dataset, "metodo": method}
        for dataset, method in stats["split_methods"].items()
    ]
    quality = stats["quality"]
    quality_rows = [{"metrica": key, "valor": value} for key, value in quality.items()]

    return "\n".join(
        [
            "---",
            "tipo: dataset",
            "status: gerado",
            "tags:",
            "  - dados",
            "  - split",
            "  - estatisticas",
            "---",
            "",
            "# Estatisticas dos Dados Processados e Splitados",
            "",
            f"Gerado em: {stats['generated_at']}",
            "",
            "## Configuracao",
            "",
            f"- Seed: {stats['seed']}",
            "- Proporcao alvo para splits novos: 70% treino, 15% validacao, 15% teste.",
            "- Split oficial preservado quando ja havia `train`, `val` e `test` no manifesto.",
            f"- Manifesto splitado: `{split_manifest.as_posix()}`",
            f"- Arquivos de split: `{split_dir.as_posix()}`",
            f"- Datasets por cenario: `{(split_dir / 'scenarios').as_posix()}`",
            f"- Mapa operacional dos experimentos: `{(split_dir / 'experiment_datasets.csv').as_posix()}`",
            f"- Estatisticas em JSON: `{stats_json.as_posix()}`",
            "",
            "## Metodo por dataset",
            "",
            dataframe_table(methods, ["dataset", "metodo"]),
            "",
            "## Amostras por dataset",
            "",
            dataframe_table(stats["samples_by_dataset"], ["dataset", "samples"]),
            "",
            "## Amostras por split",
            "",
            dataframe_table(stats["samples_by_split"], ["split", "samples"]),
            "",
            "## Amostras por dataset e split",
            "",
            dataframe_table(
                stats["samples_by_dataset_split"],
                ["dataset", "split", "samples"],
            ),
            "",
            "## Datasets por cenario experimental",
            "",
            dataframe_table(
                stats["scenario_datasets"],
                ["scenario", "split", "samples", "path"],
            ),
            "",
            "## Matriz operacional dos experimentos",
            "",
            dataframe_table(
                stats["experiment_datasets"],
                [
                    "id",
                    "model",
                    "train_dataset",
                    "test_dataset",
                    "train_manifest",
                    "val_manifest",
                    "test_manifest",
                ],
            ),
            "",
            "## Comprimento dos textos normalizados",
            "",
            dataframe_table(
                stats["text_lengths_by_dataset"],
                [
                    "dataset",
                    "chars_mean",
                    "chars_median",
                    "chars_min",
                    "chars_max",
                    "words_mean",
                    "words_median",
                    "words_min",
                    "words_max",
                ],
            ),
            "",
            "## Resolucao das imagens processadas",
            "",
            dataframe_table(
                stats["processed_resolution_counts"],
                ["resolution", "samples"],
            ),
            "",
            "## Controle de qualidade",
            "",
            dataframe_table(quality_rows, ["metrica", "valor"]),
            "",
            "## Frequencia de caracteres",
            "",
            dataframe_table(stats["top_characters"], ["character", "count"]),
            "",
        ]
    )


def write_markdown_report(markdown: str, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = project_root_from_this_file()
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val/test splits and dataset statistics."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=project_root / "data" / "processed" / "WritingProcessed.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root / "data" / "processed" / "WritingProcessed_split.csv",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=project_root / "data" / "splits",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=project_root / "outputs" / "reports" / "split_statistics.json",
    )
    parser.add_argument(
        "--stats-md",
        type=Path,
        default=project_root / "outputs" / "reports" / "split_statistics.md",
    )
    parser.add_argument(
        "--obsidian-md",
        type=Path,
        default=project_root.parent
        / "ImagensMedicas"
        / "03-Datasets"
        / "Estatisticas-dos-Splits.md",
    )
    parser.add_argument(
        "--experiment-matrix",
        type=Path,
        default=project_root / "configs" / "experiments" / "matrix.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "data"

    manifest = read_manifest(args.input_manifest.resolve())
    split_df, methods = create_splits(manifest, seed=args.seed)
    split_df, image_path_stats = repair_image_paths(split_df, data_root)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output_manifest, index=False)
    write_split_files(split_df, args.splits_dir)
    scenario_records, scenario_lookup = write_scenario_files(
        split_df,
        args.splits_dir,
        project_root,
    )
    experiment_records = write_experiment_dataset_map(
        read_experiment_matrix(args.experiment_matrix),
        scenario_lookup,
        args.splits_dir,
    )

    stats = build_statistics(
        split_df,
        methods=methods,
        image_path_stats=image_path_stats,
        seed=args.seed,
        scenario_records=scenario_records,
        experiment_records=experiment_records,
    )
    write_statistics_json(stats, args.stats_json)

    report = build_markdown_report(
        stats=stats,
        split_manifest=args.output_manifest.relative_to(project_root),
        split_dir=args.splits_dir.relative_to(project_root),
        stats_json=args.stats_json.relative_to(project_root),
    )
    write_markdown_report(report, args.stats_md, args.obsidian_md)

    print(f"Split manifest: {args.output_manifest}")
    print(f"Split CSVs: {args.splits_dir}")
    print(f"Statistics JSON: {args.stats_json}")
    print(f"Statistics Markdown: {args.stats_md}")
    print(f"Obsidian note: {args.obsidian_md}")
    print(f"Total samples: {len(split_df)}")


if __name__ == "__main__":
    main()
