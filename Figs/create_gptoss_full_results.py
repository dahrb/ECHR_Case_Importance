"""
Publication-ready GPT-OSS full results figure.
Shows: Majority baseline, zero-shot/few-shot baselines,
Experiment 1 retrieval (BM25/FAISS/TGN ± rerank, k=3/5/10),
and Experiment 2 iterative (FAISS k=3/5/10 where complete).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ARTS = [3, 6, 8]

CASES = {
    a: pd.read_pickle(ROOT / f"data/processed/article{a}/splits/comm_test.pkl").set_index("Filename")
    for a in ARTS
}

RUNS: dict = {}


def metrics(y, p):
    src = float(spearmanr(y, p).statistic) if len(set(p)) > 1 else np.nan
    return dict(mae=float(np.mean(abs(y - p))), src=src)


def register(a, model, family, method, k, paths):
    paths = [ROOT / f"data/results/article{a}" / p for p in paths]
    path = next((p for p in paths if p.exists()), paths[0]) if paths else None
    key = (a, model, family, method, k)
    c = CASES[a]
    first = {}
    badjson = mismatch = 0
    if path and path.exists():
        for line in path.read_bytes().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                badjson += 1
                continue
            fn, p = str(r.get("Filename")), r.get("prediction")
            if type(p) is not int or p not in (1, 2, 3, 4):
                continue
            first.setdefault(fn, p)
    if family == "Majority":
        first = {fn: 4 for fn in c.index}
    valid = len(set(first) & set(c.index))
    extra = len(set(first) - set(c.index))
    complete = valid == len(c) and extra == 0 and badjson == 0
    if complete or (family == "Baseline" and valid and extra == 0):
        index = [fn for fn in c.index if fn in first]
        y = c.loc[index, "importance"].to_numpy(int)
        p = np.array([first[fn] for fn in index])
        row = dict(article=a, model=model, family=family, method=method, k=k,
                   valid=valid, target=len(c))
        row.update(metrics(y, p))
        RUNS[key] = row


def collect():
    for a in ARTS:
        register(a, "Majority", "Majority", "Zero-shot", 0, [])
        for model, tag in [("GPT-OSS", "gpt-oss-120b")]:
            base = tag
            for method, prefix in [("Zero-shot", "base"), ("Few-shot", "few_shot")]:
                register(a, model, "Baseline", method, 0,
                         [f"{prefix}_text1_test_{base}.jsonl"])
            for retr in ["faiss", "bm25", "tgn_kg"]:
                for rr in [False, True]:
                    for k in [3, 5, 10]:
                        stem = (f"retrieval_{retr}_k{k}"
                                + ("_rerank" if rr else "")
                                + f"_text1_test_{tag}")
                        paths = [stem + "_static_rerun_20260925.jsonl"]
                        if not rr or (a == 3 and retr == "tgn_kg"):
                            paths.append(stem + ".jsonl")
                        register(a, model, "Experiment 1",
                                 retr + ("+RR" if rr else ""), k, paths)
            for k in [0, 3, 5, 10]:
                stem = ("iterative_text1_test_"
                        + (f"faiss_k{k}_" if k else "")
                        + "gpt-oss-120b-1h100.jsonl")
                register(a, "GPT-OSS", "Iterative",
                         "faiss" if k else "Zero-shot", k, [stem])


collect()


# ── colour/style palette ────────────────────────────────────────────────────
FAISS_C  = "#2166ac"   # blue
TGN_C    = "#d6604d"   # red-orange
BM25_C   = "#4dac26"   # green
ITER_C   = "#762a83"   # purple
BASE_ZEROSHOT_C = "#666666"
BASE_FEWSHOT_C  = "#999999"
MAJORITY_C      = "#bbbbbb"

EXP1_METHODS = [
    ("faiss",     FAISS_C, "-",  "FAISS",     "o"),
    ("faiss+RR",  FAISS_C, "--", "FAISS + RR","s"),
    ("tgn_kg",    TGN_C,   "-",  "TGN",       "o"),
    ("tgn_kg+RR", TGN_C,   "--", "TGN + RR",  "s"),
    ("bm25",      BM25_C,  "-",  "BM25",      "o"),
    ("bm25+RR",   BM25_C,  "--", "BM25 + RR", "s"),
]

K_VALS = [3, 5, 10]
METRICS = [("mae", "MAE  ↓  (lower is better)"),
           ("src", "Spearman ρ  ↑  (higher is better)")]


def get(a, family, method, k, metric):
    key = (a, "GPT-OSS", family, method, k)
    if family == "Majority":
        key = (a, "Majority", "Majority", "Zero-shot", 0)
    row = RUNS.get(key)
    if row is None:
        return np.nan
    return row.get(metric, np.nan)


def hline(ax, y, color, ls, lw, zorder=1):
    if np.isfinite(y):
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=zorder)


# ── figure layout ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 3,
    figsize=(13, 7.8),
    sharex=True,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
})

for col, a in enumerate(ARTS):
    for row_i, (metric, ylabel) in enumerate(METRICS):
        ax = axes[row_i, col]

        # ── reference lines ──────────────────────────────────────────────────
        maj = get(a, "Majority", "Zero-shot", 0, metric)
        zs  = get(a, "Baseline", "Zero-shot", 0, metric)
        fs  = get(a, "Baseline", "Few-shot",  0, metric)
        iz  = get(a, "Iterative", "Zero-shot", 0, metric)

        hline(ax, maj, MAJORITY_C,     "-",  1.0, zorder=1)
        hline(ax, zs,  BASE_ZEROSHOT_C, ":", 1.5, zorder=2)
        hline(ax, fs,  BASE_FEWSHOT_C,  "-.", 1.5, zorder=2)
        hline(ax, iz,  ITER_C,          ":", 1.4, zorder=2)

        # ── Experiment 1 lines ───────────────────────────────────────────────
        for (method, color, ls, label, marker) in EXP1_METHODS:
            vals = [get(a, "Experiment 1", method, k, metric) for k in K_VALS]
            if any(np.isfinite(v) for v in vals):
                ax.plot(K_VALS, vals, ls=ls, color=color, lw=1.8,
                        marker=marker, ms=5.5, zorder=4)

        # ── Iterative FAISS ──────────────────────────────────────────────────
        iter_vals = [get(a, "Iterative", "faiss", k, metric) for k in K_VALS]
        if any(np.isfinite(v) for v in iter_vals):
            ax.plot(K_VALS, iter_vals, ls="-", color=ITER_C, lw=2.2,
                    marker="D", ms=5.5, zorder=5, label="Iterative FAISS")

        ax.set_xticks(K_VALS)
        ax.set_xticklabels([f"k={k}" for k in K_VALS])

        if col == 0:
            ax.set_ylabel(ylabel, labelpad=6)
        if row_i == 0:
            ax.set_title(f"Article {a}", fontweight="bold", pad=8)
        if row_i == 1:
            ax.set_xlabel("Retrieved examples", labelpad=5)

        # y-axis limits with a little padding
        if metric == "mae":
            ax.set_ylim(0.08, 1.12)
        else:
            ax.set_ylim(-0.02, 0.30)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")


# ── shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    # Baselines
    mlines.Line2D([], [], color=MAJORITY_C, ls="-",  lw=1.2, label="Majority class"),
    mlines.Line2D([], [], color=BASE_ZEROSHOT_C, ls=":", lw=1.5, label="Zero-shot baseline"),
    mlines.Line2D([], [], color=BASE_FEWSHOT_C,  ls="-.",lw=1.5, label="Few-shot baseline"),
    mlines.Line2D([], [], color=ITER_C, ls=":", lw=1.4, label="Iterative zero-shot"),
    # Exp1
    mlines.Line2D([], [], color=FAISS_C, ls="-",  lw=1.8, marker="o", ms=5, label="FAISS"),
    mlines.Line2D([], [], color=FAISS_C, ls="--", lw=1.8, marker="s", ms=5, label="FAISS + rerank"),
    mlines.Line2D([], [], color=TGN_C,   ls="-",  lw=1.8, marker="o", ms=5, label="TGN"),
    mlines.Line2D([], [], color=TGN_C,   ls="--", lw=1.8, marker="s", ms=5, label="TGN + rerank"),
    mlines.Line2D([], [], color=BM25_C,  ls="-",  lw=1.8, marker="o", ms=5, label="BM25"),
    mlines.Line2D([], [], color=BM25_C,  ls="--", lw=1.8, marker="s", ms=5, label="BM25 + rerank"),
    # Exp2
    mlines.Line2D([], [], color=ITER_C, ls="-", lw=2.2, marker="D", ms=5, label="Iterative FAISS (Exp. 2)"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=8.5,
    bbox_to_anchor=(0.5, -0.01),
    columnspacing=1.2,
    handlelength=2.2,
)

fig.tight_layout(rect=(0, 0.10, 1, 1))
out_path = OUT / "gptoss_full_results.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_path}")
