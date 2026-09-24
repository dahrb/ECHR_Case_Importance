"""
Prepare training data for LegalBERT cross-encoder re-ranker.

For each article:
  - Query: Subject Matter text from comm_cases.pkl
  - Positive: 200-word summary of a gold-matched outcome case
  - Negative: 200-word summary of a FAISS-retrieved outcome case (hard negative,
              skipping any cases that appear in the gold set)

FAISS results store outcome cases as appno strings; gold_results and
outcome_summaries use Filename strings. This script builds the appno→Filename
bridge via outcome_cases.pkl.

Output: data/vectordb/article{N}/rerank_training_data.pkl
  DataFrame columns: ['Filename', 'Comm_Case', 'Positive', 'Negative']
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_training_data(article: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    data_dir = ROOT / "data"
    processed_dir = data_dir / "processed" / f"article{article}"
    vdb_dir = data_dir / "vectordb" / f"article{article}"

    comm_cases = pd.read_pickle(processed_dir / "comm_cases.pkl")
    comm_train = pd.read_pickle(processed_dir / "splits" / "comm_train.pkl")
    comm_train_ids = set(comm_train["Filename"])
    outcome_cases = pd.read_pickle(processed_dir / "outcome_cases.pkl")
    outcome_summaries = pd.read_pickle(processed_dir / "outcome_summaries.pkl")
    gold_results_all: dict = load_pickle(vdb_dir / "gold_results.pkl")
    # Restrict to comm_train cases only — prevents reranker from seeing test queries
    gold_results = {k: v for k, v in gold_results_all.items() if k in comm_train_ids}
    faiss_results: dict = load_pickle(vdb_dir / "cosine_qwen3-8b_raw_chunk_2048_results.pkl")

    # Build appno → Filename mapping (FAISS values are appnos; gold/summaries use Filenames)
    # appno can be multi-value "a;b;c" — take first token, deduplicate keeping last
    outcome_cases = outcome_cases.copy()
    outcome_cases["_appno_first"] = outcome_cases["appno"].str.split(";").str[0].str.strip()
    outcome_cases.drop_duplicates(subset=["_appno_first"], keep="last", inplace=True)
    appno_to_file = dict(zip(outcome_cases["_appno_first"], outcome_cases["File"]))

    # Index: outcome Filename → 200-word summary
    summary_map = dict(zip(outcome_summaries["Filename"], outcome_summaries["200 Word Summary"]))

    # Index: comm Filename → Subject Matter text
    comm_text_map = dict(zip(comm_cases["Filename"], comm_cases["Subject Matter"]))

    rows = []
    skipped_no_positive_summary = 0
    skipped_no_negative = 0

    for comm_filename, gold_outcomes in gold_results.items():
        comm_text = comm_text_map.get(comm_filename)
        if not comm_text or not str(comm_text).strip():
            continue

        gold_set = set(gold_outcomes)  # Filename-format set

        # Collect positives that have summaries (gold_outcomes are Filenames)
        positives = [
            (oc, summary_map[oc])
            for oc in gold_outcomes
            if oc in summary_map and summary_map[oc]
        ]
        if not positives:
            skipped_no_positive_summary += 1
            continue

        # Collect hard negatives: FAISS results not in gold set, mapped appno→Filename
        faiss_appnos = faiss_results.get(comm_filename, [])
        hard_negatives = []
        # Skip top-15 (likely true positives even if not in gold_results)
        for raw in faiss_appnos[15:]:
            appno = raw.split(";")[0].strip()
            fname = appno_to_file.get(appno)
            if fname and fname not in gold_set and fname in summary_map and summary_map[fname]:
                hard_negatives.append(fname)

        if not hard_negatives:
            skipped_no_negative += 1
            continue

        # Emit one row per positive, cycling through hard negatives
        for i, (pos_filename, pos_summary) in enumerate(positives):
            neg_filename = hard_negatives[i % len(hard_negatives)]
            rows.append({
                "Filename": comm_filename,
                "Comm_Case": str(comm_text).strip(),
                "Positive": str(pos_summary).strip(),
                "Negative": str(summary_map[neg_filename]).strip(),
            })

    df = pd.DataFrame(rows, columns=["Filename", "Comm_Case", "Positive", "Negative"])

    print(f"Article {article}: {len(df)} training pairs from {len(gold_results)} comm_train cases (of {len(gold_results_all)} total)")
    print(f"  skipped (no positive summary): {skipped_no_positive_summary}")
    print(f"  skipped (no hard negative): {skipped_no_negative}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", type=int, required=True, choices=[3, 6, 8],
                        help="ECHR article number")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_training_data(args.article, seed=args.seed)

    if df.empty:
        print("No training data generated — check that summaries are available.", file=sys.stderr)
        sys.exit(1)

    out_path = ROOT / "data" / "vectordb" / f"article{args.article}" / "rerank_training_data.pkl"
    df.to_pickle(out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
