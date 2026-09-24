"""Create communicated-case label distribution figures for Articles 3, 6, and 8.

The figures use the current experimental test splits in
``data/processed/article{article}/splits/comm_test.pkl``. Importance values are
shown using the labels exposed to the models, while court levels use the same
source-file mapping as ``echr.prediction.prompts.COURT_SOURCE_MAP``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent
ARTICLES = (3, 6, 8)

IMPORTANCE_LABELS = {
    1: "Key case",
    2: "Level 1",
    3: "Level 2",
    4: "Level 3",
}
COURT_SOURCE_LABELS = {
    "pruned_COMMITTEE_meta.json": "Committee",
    "pruned_ADMISSIBILITYCOM_meta.json": "Committee",
    "pruned_CHAMBER_meta.json": "Chamber",
    "pruned_ADMISSIBILITY_meta.json": "Chamber",
    "pruned_GRANDCHAMBER_meta.json": "Grand Chamber",
    "pruned_DECGRANDCHAMBER_meta.json": "Grand Chamber",
}

ARTICLE_COLOURS = {
    3: "#3B6FB6",
    6: "#D17A22",
    8: "#3A9D78",
}


def load_test_cases() -> dict[int, pd.DataFrame]:
    """Load and validate the current communicated-case experimental test sets."""
    cases = {}
    for article in ARTICLES:
        path = (
            ROOT
            / "data"
            / "processed"
            / f"article{article}"
            / "splits"
            / "comm_test.pkl"
        )
        frame = pd.read_pickle(path)
        required = {"Filename", "importance", "source_file"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        if frame["Filename"].duplicated().any():
            raise ValueError(f"{path} contains duplicate communicated-case filenames")
        cases[article] = frame
    return cases


def build_summary(cases: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Return tidy counts and percentages for both label types."""
    records = []
    for article, frame in cases.items():
        importance = frame["importance"].map(IMPORTANCE_LABELS)
        if importance.isna().any():
            unknown = sorted(frame.loc[importance.isna(), "importance"].unique())
            raise ValueError(f"Unknown importance values for Article {article}: {unknown}")

        court = frame["source_file"].map(COURT_SOURCE_LABELS)
        if court.isna().any():
            unknown = sorted(frame.loc[court.isna(), "source_file"].unique())
            raise ValueError(f"Unknown court sources for Article {article}: {unknown}")

        for variable, values, order in (
            ("Importance", importance, list(IMPORTANCE_LABELS.values())),
            ("Court level", court, ["Grand Chamber", "Chamber", "Committee"]),
        ):
            counts = values.value_counts().reindex(order, fill_value=0)
            for label, count in counts.items():
                records.append(
                    {
                        "article": article,
                        "variable": variable,
                        "label": label,
                        "count": int(count),
                        "percent": 100 * count / len(frame),
                        "total_cases": len(frame),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_distribution(
    summary: pd.DataFrame,
    variable: str,
    order: list[str],
    output_stem: str,
) -> None:
    """Plot one count-distribution panel per article with percentage annotations."""
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    subset = summary[summary["variable"] == variable]
    ymax = subset["count"].max() * 1.18

    for axis, article in zip(axes, ARTICLES):
        article_data = (
            subset[subset["article"] == article]
            .set_index("label")
            .reindex(order)
        )
        bars = axis.bar(
            article_data.index,
            article_data["count"],
            color=ARTICLE_COLOURS[article],
            edgecolor="#222222",
            linewidth=0.8,
            width=0.72,
        )
        total = int(article_data["total_cases"].iloc[0])
        axis.set_title(f"Article {article} (n = {total:,})", fontweight="bold")
        axis.set_xlabel(variable)
        axis.set_ylim(0, ymax)
        axis.tick_params(axis="x", rotation=18)

        for bar, count, percent in zip(
            bars, article_data["count"], article_data["percent"]
        ):
            axis.annotate(
                f"{int(count):,}\n({percent:.1f}%)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[0].set_ylabel("Number of communicated cases")
    figure.tight_layout()
    figure.savefig(
        FIG_DIR / f"{output_stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    cases = load_test_cases()
    summary = build_summary(cases)

    plot_distribution(
        summary,
        variable="Importance",
        order=list(IMPORTANCE_LABELS.values()),
        output_stem="importance_distribution",
    )
    plot_distribution(
        summary,
        variable="Court level",
        order=["Grand Chamber", "Chamber", "Committee"],
        output_stem="court_level_distribution",
    )


if __name__ == "__main__":
    main()
