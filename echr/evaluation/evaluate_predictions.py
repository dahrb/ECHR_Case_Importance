"""
Evaluate and compare prediction results across conditions, articles, and models.

Usage:
    python echr/evaluation/evaluate_predictions.py
    python echr/evaluation/evaluate_predictions.py --article 3
    python echr/evaluation/evaluate_predictions.py --article 3 --split test

Output: table to stdout (and optionally CSV)
"""

import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def evaluate_file(path: str) -> dict:
    rows = load_jsonl(path)
    if not rows:
        return None

    valid = [r for r in rows if r.get("prediction") is not None and r.get("importance") is not None]
    if not valid:
        return None

    true = [r["importance"] for r in valid]
    pred = [r["prediction"] for r in valid]
    n_total = len(rows)
    n_valid = len(valid)
    n_null = n_total - n_valid

    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    acc = accuracy_score(true, pred)

    # per-class F1
    labels = sorted(set(true) | set(pred))
    per_class = f1_score(true, pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = {f"f1_imp{labels[i]}": round(float(per_class[i]), 3) for i in range(len(labels))}

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_null": n_null,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "pred_dist": dict(sorted(Counter(pred).items())),
        "true_dist": dict(sorted(Counter(true).items())),
        **per_class_f1,
    }


def parse_filename(fname: str) -> dict:
    """Extract metadata from result filename."""
    base = os.path.basename(fname).replace(".jsonl", "")
    parts = base.split("_")
    # filename format: {condition}[_cot][_text{N}]_{split}_{model}
    # find split (test or valid)
    split = "test" if "test" in parts else "valid"
    split_idx = parts.index(split)
    model = "_".join(parts[split_idx + 1:])

    # text field
    text = "1"
    text_part = [p for p in parts if p.startswith("text")]
    if text_part:
        text = text_part[0].replace("text", "")

    # condition
    cond_parts = [p for p in parts[:split_idx] if not p.startswith("text")]
    condition = "_".join(cond_parts)

    return {"condition": condition, "text": text, "split": split, "model": model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default=None, help="Filter by article (3, 6, 8, or all)")
    parser.add_argument("--split", default="test", help="test or valid")
    parser.add_argument("--csv", default=None, help="Save results to CSV path")
    args = parser.parse_args()

    articles = [args.article] if args.article else ["3", "6", "8"]
    records = []

    for art in articles:
        result_dir = os.path.join(DATA, "results", f"article{art}")
        if not os.path.exists(result_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(result_dir, "*.jsonl"))):
            meta = parse_filename(fpath)
            if args.split and meta["split"] != args.split:
                continue
            metrics = evaluate_file(fpath)
            if metrics is None:
                continue
            records.append({
                "article": art,
                **meta,
                **metrics,
                "file": os.path.basename(fpath),
            })

    if not records:
        print("No results found.")
        return

    df = pd.DataFrame(records)
    display_cols = ["article", "condition", "split", "model", "n_total", "n_null",
                    "macro_f1", "accuracy"]
    per_class_cols = [c for c in df.columns if c.startswith("f1_imp")]
    display_cols += sorted(per_class_cols)

    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved to {args.csv}")


if __name__ == "__main__":
    main()
