"""Plot confusion matrices for the strongest complete Experiment 1 runs.

For each article, the left panel is the complete Experiment 1 configuration
with the lowest MAE and the right panel is the configuration with the highest
Spearman rank correlation (SRC). Selection uses the current Experiment 1
matrix. Heatmap colours are normalized within each true class; annotations
show the count and row percentage.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent
LABELS = (1, 2, 3, 4)
LABEL_NAMES = {
    1: "1\nKey case",
    2: "2\nLevel 1",
    3: "3\nLevel 2",
    4: "4\nLevel 3",
}

# Selected over all complete current Experiment 1 cells (BM25, FAISS, and TGN;
# k=3/5/10; with and without reranking; both models).
SELECTIONS = {
    3: (
        {
            "criterion": "Lowest MAE",
            "configuration": "FAISS, k=10, no rerank",
            "mae": 0.3964,
            "src": 0.2112,
            "filename": (
                "retrieval_faiss_k10_text1_test_"
                "gpt-oss-120b_static_rerun_20260925.jsonl"
            ),
        },
        {
            "criterion": "Highest SRC",
            "configuration": "FAISS, k=3, no rerank",
            "mae": 0.5291,
            "src": 0.2242,
            "filename": (
                "retrieval_faiss_k3_text1_test_"
                "gpt-oss-120b_static_rerun_20260925.jsonl"
            ),
        },
    ),
    6: (
        {
            "criterion": "Lowest MAE",
            "configuration": "TGN, k=10, no rerank",
            "mae": 0.3896,
            "src": 0.0623,
            "filename": "retrieval_tgn_kg_k10_text1_test_gpt-oss-120b.jsonl",
        },
        {
            "criterion": "Highest SRC",
            "configuration": "FAISS, k=3, no rerank",
            "mae": 0.4426,
            "src": 0.1711,
            "filename": (
                "retrieval_faiss_k3_text1_test_"
                "gpt-oss-120b_static_rerun_20260925.jsonl"
            ),
        },
    ),
    8: (
        {
            "criterion": "Lowest MAE",
            "configuration": "FAISS, k=10, no rerank",
            "mae": 0.5673,
            "src": 0.2073,
            "filename": (
                "retrieval_faiss_k10_text1_test_"
                "gpt-oss-120b_static_rerun_20260925.jsonl"
            ),
        },
        {
            "criterion": "Highest SRC",
            "configuration": "FAISS, k=3, reranked",
            "mae": 0.7315,
            "src": 0.2465,
            "filename": (
                "retrieval_faiss_k3_rerank_text1_test_"
                "gpt-oss-120b_static_rerun_20260925.jsonl"
            ),
        },
    ),
}


def load_test_cases(article: int) -> pd.DataFrame:
    path = (
        ROOT
        / "data"
        / "processed"
        / f"article{article}"
        / "splits"
        / "comm_test.pkl"
    )
    frame = pd.read_pickle(path)[["Filename", "importance"]].copy()
    if frame["Filename"].duplicated().any():
        raise ValueError(f"Duplicate filenames in {path}")
    frame["Filename"] = frame["Filename"].astype(str)
    frame["importance"] = frame["importance"].astype(int)
    observed = set(frame["importance"])
    if not observed.issubset(LABELS):
        raise ValueError(f"Unexpected importance labels in {path}: {observed}")
    return frame


def load_valid_predictions(path: Path) -> dict[str, int]:
    """Load one prediction per case, preferring the latest valid retry."""
    predictions: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            filename = row.get("Filename")
            prediction = row.get("prediction")
            if filename is None or prediction is None:
                continue
            prediction = int(prediction)
            if prediction not in LABELS:
                raise ValueError(
                    f"Unexpected prediction {prediction} in {path}:{line_number}"
                )
            predictions[str(filename)] = prediction
    return predictions


def build_matrix(
    cases: pd.DataFrame,
    predictions: dict[str, int],
    source: Path,
) -> tuple[np.ndarray, int]:
    expected = set(cases["Filename"])
    unexpected = set(predictions).difference(expected)
    if unexpected:
        raise ValueError(
            f"{source} contains {len(unexpected)} cases outside the test split"
        )
    joined = cases.copy()
    joined["prediction"] = joined["Filename"].map(predictions)
    valid = joined.dropna(subset=["prediction"]).copy()
    valid["prediction"] = valid["prediction"].astype(int)
    matrix = np.zeros((4, 4), dtype=int)
    for truth, prediction in valid[["importance", "prediction"]].itertuples(
        index=False, name=None
    ):
        matrix[LABELS.index(int(truth)), LABELS.index(int(prediction))] += 1
    return matrix, len(valid)


def row_percentages(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix * 100.0,
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals != 0,
    )


def annotation_colour(percent: float) -> str:
    return "white" if percent >= 52 else "#151515"


def write_counts(records: list[dict[str, object]]) -> None:
    fields = [
        "article",
        "selection_criterion",
        "model",
        "configuration",
        "mae",
        "src",
        "true_label",
        "predicted_label",
        "count",
        "row_percent",
        "valid_predictions",
        "test_cases",
        "source_file",
    ]
    with (FIG_DIR / "experiment1_confusion_matrix_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(9.2, 12.3), sharex=True, sharey=True)
    records: list[dict[str, object]] = []
    image = None

    for row_index, (article, selections) in enumerate(SELECTIONS.items()):
        cases = load_test_cases(article)
        total_cases = len(cases)
        for column_index, selection in enumerate(selections):
            axis = axes[row_index, column_index]
            source = (
                ROOT
                / "data"
                / "results"
                / f"article{article}"
                / str(selection["filename"])
            )
            predictions = load_valid_predictions(source)
            matrix, valid_count = build_matrix(cases, predictions, source)
            if valid_count != total_cases:
                raise ValueError(
                    f"Selected run is incomplete: {source} ({valid_count}/{total_cases})"
                )
            percentages = row_percentages(matrix)
            image = axis.imshow(
                percentages,
                cmap="Blues",
                vmin=0,
                vmax=100,
                interpolation="nearest",
                aspect="equal",
            )
            for true_index, true_label in enumerate(LABELS):
                for predicted_index, predicted_label in enumerate(LABELS):
                    count = int(matrix[true_index, predicted_index])
                    percent = float(percentages[true_index, predicted_index])
                    axis.text(
                        predicted_index,
                        true_index,
                        f"{count:,}\n{percent:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7.7,
                        color=annotation_colour(percent),
                    )
                    records.append(
                        {
                            "article": article,
                            "selection_criterion": selection["criterion"],
                            "model": "GPT-OSS-120B",
                            "configuration": selection["configuration"],
                            "mae": selection["mae"],
                            "src": selection["src"],
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": count,
                            "row_percent": f"{percent:.6f}",
                            "valid_predictions": valid_count,
                            "test_cases": total_cases,
                            "source_file": source.relative_to(ROOT),
                        }
                    )
            axis.set_title(
                f"{selection['criterion']}: {selection['configuration']}\n"
                f"MAE = {selection['mae']:.3f}, SRC = {selection['src']:.3f}",
                fontweight="bold",
                pad=9,
            )
            axis.set_xticks(range(4), [LABEL_NAMES[label] for label in LABELS])
            axis.set_yticks(range(4), [LABEL_NAMES[label] for label in LABELS])
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_color("#333333")
                spine.set_linewidth(0.7)
            if column_index == 0:
                axis.set_ylabel(
                    f"Article {article}\nTrue class",
                    fontweight="bold",
                    labelpad=9,
                )
            if row_index == len(SELECTIONS) - 1:
                axis.set_xlabel("Predicted class", labelpad=7)

    if image is None:
        raise RuntimeError("No confusion matrices were plotted")
    figure.subplots_adjust(
        left=0.11,
        right=0.97,
        bottom=0.07,
        top=0.96,
        wspace=0.16,
        hspace=0.25,
    )
    figure.savefig(
        FIG_DIR / "experiment1_best_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    write_counts(records)


if __name__ == "__main__":
    main()
