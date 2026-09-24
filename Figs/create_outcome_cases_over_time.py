"""Plot annual outcome-case counts for Articles 3, 6, and 8."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent
ARTICLES = (3, 6, 8)

SERIES_STYLE = {
    3: {"color": "#3B6FB6", "marker": "o"},
    6: {"color": "#D17A22", "marker": "s"},
    8: {"color": "#3A9D78", "marker": "^"},
}


def annual_outcome_counts() -> pd.DataFrame:
    """Calculate annual counts directly from the processed outcome-case files."""
    series = {}
    for article in ARTICLES:
        path = ROOT / "data" / "processed" / f"article{article}" / "outcome_cases.pkl"
        cases = pd.read_pickle(path)

        required = {"File", "date"}
        missing = required.difference(cases.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        if cases["File"].duplicated().any():
            raise ValueError(f"{path} contains duplicate outcome-case identifiers")

        dates = pd.to_datetime(cases["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"{path} contains {int(dates.isna().sum())} invalid dates")
        series[article] = dates.dt.year.value_counts().sort_index()

    first_year = min(counts.index.min() for counts in series.values())
    last_year = max(counts.index.max() for counts in series.values())
    years = pd.Index(range(first_year, last_year + 1), name="year")
    return pd.DataFrame(
        {article: counts.reindex(years, fill_value=0) for article, counts in series.items()},
        index=years,
    )


def plot_counts(counts: pd.DataFrame) -> None:
    """Render one line per article using the repository's publication style."""
    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    for article in ARTICLES:
        style = SERIES_STYLE[article]
        axis.plot(
            counts.index,
            counts[article],
            label=f"Article {article}",
            color=style["color"],
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.2,
        )

    first_year = int(counts.index.min())
    last_year = int(counts.index.max())
    ticks = list(range(first_year, last_year + 1, 5))
    if last_year not in ticks:
        ticks.append(last_year)

    axis.set_xlabel("Year")
    axis.set_ylabel("Number of outcome cases")
    axis.set_xticks(ticks)
    axis.set_xlim(first_year - 0.5, last_year + 0.5)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_facecolor("white")
    figure.set_facecolor("white")
    figure.tight_layout()
    figure.savefig(
        FIG_DIR / "outcome_cases_over_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({"font.size": 12})
    plot_counts(annual_outcome_counts())


if __name__ == "__main__":
    main()
