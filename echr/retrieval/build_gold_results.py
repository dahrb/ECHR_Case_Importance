"""
Build GOLD retrieval results using HUDOC extractedappno metadata.

For each comm_test case the "gold" retrieved cases are the outcome cases whose
application numbers appear in the HUDOC extractedappno field — the cases that
the ECtHR system explicitly links to this communicated case.

Source: data/processed/metadata/unfiltered_metadata.pkl (doctypebranch=COMMUNICATEDCASES)
All three articles have full coverage (1324/1324 Art3, 2451/2451 Art6, 1354/1354 Art8).

Output: data/vectordb/article{N}/gold_results.pkl
    Format: {comm_Filename: [outcome_File, ...]} — same as bm25/faiss results.
"""

import argparse
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
KEEP = 100


def build_gold(article: str) -> None:
    comm = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "splits", "comm_test.pkl")
    )
    oc = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "outcome_cases.pkl")
    )

    # Load HUDOC metadata for comm cases — has extractedappno for all articles
    meta_path = os.path.join(DATA, "processed", "metadata", "unfiltered_metadata.pkl")
    meta = pd.read_pickle(meta_path)
    comm_meta = meta[meta["doctypebranch"] == "COMMUNICATEDCASES"].set_index("itemid")

    # Build appno → File mapping for outcome cases
    appno_to_rows = {}
    for _, row in oc.iterrows():
        for app in str(row["appno"]).split(";"):
            key = app.strip()
            appno_to_rows.setdefault(key, []).append(row)

    results = {}
    matched_total = 0
    for _, row in comm.iterrows():
        fn = row["Filename"]
        # Get extractedappno from HUDOC metadata
        meta_row = comm_meta.loc[fn] if fn in comm_meta.index else None
        extracted_raw = meta_row["extractedappno"] if meta_row is not None else None

        extracted = []
        if extracted_raw and str(extracted_raw) not in ("nan", "None", ""):
            extracted = [a.strip() for a in str(extracted_raw).split(";") if a.strip()]

        ranked = []
        for app in extracted:
            first = app.split(";")[0].strip()
            rows = appno_to_rows.get(first, [])
            if rows:
                best = sorted(rows, key=lambda r: r["date"])[-1]
                file_id = best["File"]
                if file_id not in ranked:
                    ranked.append(file_id)

        results[fn] = ranked[:KEEP]
        if ranked:
            matched_total += 1

    out_dir = os.path.join(DATA, "vectordb", f"article{article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gold_results.pkl")
    pd.to_pickle(results, out_path)

    avg_k = sum(len(v) for v in results.values()) / len(results) if results else 0
    print(f"[art {article}] gold_results: {len(results)} cases, "
          f"{matched_total} with links, avg {avg_k:.1f} docs/case")
    print(f"[art {article}] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="3, 6, or 8")
    args = parser.parse_args()
    build_gold(args.article)


if __name__ == "__main__":
    main()
