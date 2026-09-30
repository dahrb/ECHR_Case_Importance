"""Create an Article-3-only figure of zero-shot and few-shot confusion matrices.

A single-row variant of ``create_baseline_confusion_matrices.py`` restricted to
Article 3. Columns are the GPT-OSS-120B and Llama-3.3-70B-Instruct FP8
zero-shot/few-shot baselines. Heatmap colours show percentages normalized within
each true class; annotations report the underlying count and percentage. Null
predictions are excluded and each panel reports valid coverage against the full
communicated-case test split.
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
ARTICLE = 3
LABELS = (1, 2, 3, 4)
LABEL_NAMES = {
    1: "1\nKey case",
    2: "2\nLevel 1",
    3: "3\nLevel 2",
    4: "4\nLevel 3",
}

CONDITIONS = (
    {
        "model": "GPT-OSS-120B",
        "condition": "Zero-shot",
        "filename": "base_text1_test_gpt-oss-120b.jsonl",
    },
    {
        "model": "GPT-OSS-120B",
        "condition": "Few-shot",
        "filename": "few_shot_text1_test_gpt-oss-120b.jsonl",
    },
    {
        "model": "Llama-3.3-70B-Instruct (FP8)",
        "condition": "Zero-shot",
        "filename": "base_text1_test_Llama-3.3-70B-Instruct-FP8.jsonl",
    },
    {
        "model": "Llama-3.3-70B-Instruct (FP8)",
        "condition": "Few-shot",
        "filename": "few_shot_text1_test_Llama-3.3-70B-Instruct-FP8.jsonl",
    },
)


def load_test_cases(article: int) -> pd.DataFrame:
    """Load the experimental test split and validate its identifiers/labels."""
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
    observed = set(frame["importance"].dropna().astype(int))
    if not observed.issubset(LABELS):
        raise ValueError(f"Unexpected importance labels in {path}: {observed}")
    frame["importance"] = frame["importance"].astype(int)
    return frame


def load_valid_predictions(path: Path) -> dict[str, int]:
    """Return one valid prediction per filename, preferring the latest valid row."""
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


def confusion_matrix(
    cases: pd.DataFrame,
    predictions: dict[str, int],
    source: Path,
) -> tuple[np.ndarray, int]:
    """Build a fixed 1--4 matrix after checking predictions against the split."""
    expected = set(cases["Filename"].astype(str))
    unexpected = set(predictions).difference(expected)
    if unexpected:
        raise ValueError(
            f"{source} contains {len(unexpected)} filenames outside the test split"
        )

    joined = cases.copy()
    joined["Filename"] = joined["Filename"].astype(str)
    joined["prediction"] = joined["Filename"].map(predictions)
    valid = joined.dropna(subset=["prediction"]).copy()
    valid["prediction"] = valid["prediction"].astype(int)

    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for truth, prediction in valid[["importance", "prediction"]].itertuples(
        index=False, name=None
    ):
        matrix[LABELS.index(int(truth)), LABELS.index(int(prediction))] += 1
    return matrix, len(valid)


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each true-class row to percentages."""
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix * 100.0,
        totals,
        out=np.zeros_like(matrix, dtype=float),
        where=totals != 0,
    )


def annotation_colour(percent: float) -> str:
    """Choose legible annotation text for the Blues colour map."""
    return "white" if percent >= 52 else "#151515"


def write_counts(records: list[dict[str, object]]) -> None:
    """Save the plotted source values for reproducibility."""
    path = FIG_DIR / "baseline_confusion_matrix_art3_counts.csv"
    fields = [
        "article",
        "model",
        "condition",
        "true_label",
        "predicted_label",
        "count",
        "row_percent",
        "valid_predictions",
        "test_cases",
        "missing_predictions",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.4, 8.4),
        sharex=True,
        sharey=True,
    )
    records: list[dict[str, object]] = []
    image = None

    cases = load_test_cases(ARTICLE)
    total_cases = len(cases)

    for panel_index, condition in enumerate(CONDITIONS):
        row_index, column_index = divmod(panel_index, 2)
        axis = axes[row_index, column_index]
        source = (
            ROOT
            / "data"
            / "results"
            / f"article{ARTICLE}"
            / str(condition["filename"])
        )
        predictions = load_valid_predictions(source)
        matrix, valid_count = confusion_matrix(cases, predictions, source)
        percentages = normalized_rows(matrix)

        image = axis.imshow(
            percentages,
            cmap="Blues",
            vmin=0,
            vmax=100,
            interpolation="nearest",
            aspect="equal",
        )

        for true_index in range(len(LABELS)):
            for predicted_index in range(len(LABELS)):
                count = int(matrix[true_index, predicted_index])
                percent = float(percentages[true_index, predicted_index])
                axis.text(
                    predicted_index,
                    true_index,
                    f"{count:,}\n{percent:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=annotation_colour(percent),
                )
                records.append(
                    {
                        "article": ARTICLE,
                        "model": condition["model"],
                        "condition": condition["condition"],
                        "true_label": LABELS[true_index],
                        "predicted_label": LABELS[predicted_index],
                        "count": count,
                        "row_percent": f"{percent:.6f}",
                        "valid_predictions": valid_count,
                        "test_cases": total_cases,
                        "missing_predictions": total_cases - valid_count,
                        "source_file": source.relative_to(ROOT),
                    }
                )

        if row_index == 0:
            axis.set_title(
                str(condition["condition"]),
                fontweight="bold",
                pad=10,
            )

        axis.text(
            0.98,
            0.98,
            f"n = {valid_count:,}/{total_cases:,}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#777777",
                "linewidth": 0.5,
                "alpha": 0.92,
            },
        )

        axis.set_xticks(range(len(LABELS)), [LABEL_NAMES[label] for label in LABELS])
        axis.set_yticks(range(len(LABELS)), [LABEL_NAMES[label] for label in LABELS])
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.7)

        if column_index == 0:
            model_label = str(condition["model"]).replace(
                "Llama-3.3-70B-Instruct (FP8)",
                "Llama-3.3-70B-Instruct (FP8)",
            )
            axis.set_ylabel(
                f"{model_label}\nTrue class",
                fontweight="bold",
                labelpad=10,
            )
        if row_index == 1:
            axis.set_xlabel("Predicted class", labelpad=7)

    if image is None:
        raise RuntimeError("No confusion matrices were plotted")

    figure.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.09,
        top=0.90,
        wspace=0.08,
        hspace=0.12,
    )

    output = FIG_DIR / "baseline_confusion_matrices_art3.png"
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    write_counts(records)


if __name__ == "__main__":
    main()
