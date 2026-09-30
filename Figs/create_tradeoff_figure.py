"""
Combined MAE <-> SRC trade-off figure.

GPT-OSS configurations (excluding gold standard) + Llama FT overlay per Article,
plotted in (MAE, Spearman) space. Top-left corner is ideal.
Pareto frontier drawn; best-balanced configuration starred.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ARTS = [3, 6, 8]

df = pd.read_csv(ROOT / "data/results/all_metrics_dump.csv")

# ---- family/style map (GPT-OSS only, no Gold) -------------------------------
def style_for(row):
    fam, meth = row["family"], str(row["method"])
    if fam == "Majority":
        return None  # handled as vline only
    if fam == "Baseline":
        return dict(c="#888888", m="P", s=90, label="Zero/few-shot (GPT-OSS)", z=4)
    if fam == "Iterative":
        if meth == "Zero-shot":
            return dict(c="#c2a5cf", m="D", s=70, label="Iterative zero-shot", z=4)
        return dict(c="#762a83", m="D", s=95, label="Iterative FAISS", z=5)
    if fam == "Exp1":
        base = meth.replace("+RR", "")
        col = {"faiss": "#2166ac", "tgn_kg": "#d6604d", "bm25": "#4dac26"}[base]
        rr = "+RR" in meth
        return dict(c=col, m="s" if rr else "o", s=70,
                    label={"faiss": "FAISS", "tgn_kg": "TGN", "bm25": "BM25"}[base]
                          + (" + rerank" if rr else ""),
                    z=3, hollow=rr)
    return None  # skip Gold, FT-old stubs, etc.

def pareto(points):
    """Return boolean mask of Pareto-optimal points (min MAE, max SRC)."""
    pts = points.copy()
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]
                    and (pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1])):
                keep[i] = False
                break
    return keep

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.6,
})
fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.6))
# Articles 3 & 6 on top, Article 8 bottom-left, legend in bottom-right cell
POSITIONS = {3: axes[0, 0], 6: axes[0, 1], 8: axes[1, 0]}
legend_ax = axes[1, 1]
legend_ax.axis("off")

seen_labels = {}
best_rows = {}

for a in ARTS:
    ax = POSITIONS[a]
    # GPT-OSS: exclude Gold and FT stubs (valid < 95% of target)
    sub_gpt = df[
        (df.article == a) & (df.model == "GPT-OSS")
        & (~df.family.isin(["Gold", "Majority"]))
        & (df.valid >= 0.97 * df.target)
        & (~df.method.str.contains("old", na=False))
    ].copy()

    maj = df[(df.article == a) & (df.family == "Majority")].iloc[0]

    # Plot GPT-OSS with defined SRC
    plotted = sub_gpt[sub_gpt.src.notna()].copy()
    for _, r in plotted.iterrows():
        st = style_for(r)
        if st is None:
            continue
        fc = "none" if st.get("hollow") else st["c"]
        ax.scatter(r.mae, r.src, s=st["s"], marker=st["m"],
                   facecolors=fc, edgecolors=st["c"], linewidths=1.4,
                   zorder=st["z"], alpha=0.9)
        seen_labels.setdefault(st["label"],
                               (st["c"], st["m"], st.get("hollow", False)))

    # majority vertical line
    ax.axvline(maj.mae, color="#222222", ls=":", lw=1.2, zorder=1)
    ax.text(maj.mae + 0.005, 0.005, "majority\nMAE",
            rotation=90, va="bottom", ha="left", fontsize=7, color="#555555",
            transform=ax.get_xaxis_transform())

    # Pareto frontier (GPT-OSS only)
    pts = plotted[["mae", "src"]].to_numpy()
    mask = pareto(pts)
    front = plotted[mask].sort_values("mae")
    ax.plot(front.mae, front.src, color="#333333", lw=1.1, ls="-",
            alpha=0.45, zorder=2)

    # best-balanced point
    mae_n = (plotted.mae - plotted.mae.min()) / (plotted.mae.max() - plotted.mae.min())
    src_n = (plotted.src - plotted.src.min()) / (plotted.src.max() - plotted.src.min())
    dist = np.sqrt(mae_n**2 + (1 - src_n)**2)
    bi = dist.idxmin()
    b = plotted.loc[bi]
    best_rows[a] = b
    ax.scatter(b.mae, b.src, s=380, marker="*", facecolors="none",
               edgecolors="#111111", linewidths=1.6, zorder=7)
    ax.annotate(f"{b.family}:{b.method} k{int(b.k)}\nMAE={b.mae:.3f}, ρ={b.src:.3f}",
                (b.mae, b.src), textcoords="offset points", xytext=(8, 8),
                fontsize=7.8, fontweight="bold")

    ax.set_title(f"Article {a}", fontweight="bold")
    ax.set_xlabel("MAE  (← lower is better)")
    if ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("Spearman ρ  (higher is better →)")
    ax.margins(0.12)

# shared legend, placed in the empty bottom-right cell
handles = [mlines.Line2D([], [], color=c, marker=m, ls="",
                         markerfacecolor="none" if hollow else c,
                         markeredgecolor=c, markersize=9, label=lab)
           for lab, (c, m, hollow) in seen_labels.items()]
handles.append(mlines.Line2D([], [], color="#111111", marker="*", ls="",
                             markerfacecolor="none", markersize=15,
                             label="Best-balanced (per Article)"))
legend_ax.legend(handles=handles, loc="center", ncol=1, frameon=False,
                 fontsize=10, handletextpad=0.6, labelspacing=0.9)
fig.tight_layout()
p = OUT / "gptoss_mae_src_tradeoff.png"
fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved ->", p)
for a in ARTS:
    b = best_rows[a]
    print(f"Art{a} best-balanced: {b.family} {b.method} k{int(b.k)}  MAE={b.mae:.3f} SRC={b.src:.3f}")
