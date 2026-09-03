"""
Builds network_nodes.csv and network_edges.csv for a given article.

Reads outcome_cases_network.pkl (produced by process_outcome_cases.py),
which already contains pre-computed `citations` (list of unique_app_no values)
and `unique_app_no`.  Run process_outcome_cases.py first if the file is absent.

Output columns:
  network_nodes.csv : id, outcome, importance, branch, date, respondent,
                      appno, ecli, facts, the_law
  network_edges.csv : source, target  (both are File/itemid values)

Usage:
    python echr/graph/process_outcome_cases.py --article 3
    python echr/graph/build_network.py --article 3
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")


def build(article):
    pkl = os.path.join(DATA, "processed", f"article{article}", "outcome_cases_network.pkl")
    df = pd.read_pickle(pkl)
    print(f"[art {article}] loaded {len(df)} cases from outcome_cases_network.pkl")

    # --- build lookup: unique_app_no -> File (itemid) ---
    appno_to_file = dict(zip(df["unique_app_no"], df["File"]))

    # --- build edges from pre-computed citations (unique_app_no lists) ---
    edge_rows = []
    for _, row in df.iterrows():
        src = row["File"]
        for cited_appno in row["citations"]:
            tgt = appno_to_file.get(cited_appno)
            if tgt and src != tgt:
                edge_rows.append({"source": src, "target": tgt})

    total = len(edge_rows)
    edges = pd.DataFrame(edge_rows).drop_duplicates() if edge_rows else pd.DataFrame(columns=["source", "target"])
    print(f"[art {article}] total citation edges: {total}, deduplicated: {len(edges)}")

    # --- build nodes CSV ---
    nodes = df[["File", "conclusion", "importance", "doctypebranch",
                "date", "respondent", "appno"]].copy()
    nodes = nodes.rename(columns={
        "File": "id",
        "conclusion": "outcome",
        "doctypebranch": "branch",
    })
    nodes["date"] = pd.to_datetime(nodes["date"]).dt.strftime("%Y-%m-%d")
    nodes["ecli"] = ""
    nodes["facts"] = df["Facts"].values
    nodes["the_law"] = df["The Law"].values
    nodes = nodes[["id", "outcome", "importance", "branch", "date",
                   "respondent", "appno", "ecli", "facts", "the_law"]]

    # --- save ---
    out_dir = os.path.join(DATA, "network", f"article{article}")
    os.makedirs(out_dir, exist_ok=True)
    nodes.to_csv(os.path.join(out_dir, "network_nodes.csv"), index=False)
    edges.to_csv(os.path.join(out_dir, "network_edges.csv"), index=False)
    print(f"[art {article}] saved {len(nodes)} nodes -> {out_dir}/network_nodes.csv")
    print(f"[art {article}] saved {len(edges)} edges -> {out_dir}/network_edges.csv")
    return nodes, edges


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", required=True)
    args = ap.parse_args()
    build(args.article)
