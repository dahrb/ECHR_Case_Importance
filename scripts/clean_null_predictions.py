#!/usr/bin/env python3
"""Drop null/failed prediction rows from result JSONL files so --resume re-does them.

`run_*_predictions.py --resume` keys on Filename PRESENCE, not validity — a row with
prediction=null is treated as "done" and skipped. When a server was dead/degraded
(e.g. the L40S Llama outage) it wrote many null rows; those never get retried unless
we remove them here first.

A row is kept iff `prediction` parses to an int in 1..4. Everything else (null,
empty, "None", non-numeric, n_examples:0 failures) is dropped. Rewrites in place
after a .bak copy. Prints per-file kept/dropped.

Usage:
  python scripts/clean_null_predictions.py data/results/article3/*Llama-3.3-70B-Instruct-FP8.jsonl
  python scripts/clean_null_predictions.py --glob 'data/results/article*/*Instruct-FP8.jsonl'
"""
import argparse
import glob as globmod
import json
import os
import shutil
import sys


def valid_pred(v) -> bool:
    if v is None:
        return False
    try:
        return int(str(v).strip()) in (1, 2, 3, 4)
    except (ValueError, TypeError):
        return False


def clean_file(path: str, backup: bool = True) -> tuple[int, int]:
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if valid_pred(row.get("prediction")):
                kept.append(line)
            else:
                dropped += 1
    if dropped:
        if backup:
            shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            for l in kept:
                f.write(l + "\n")
    return len(kept), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="result .jsonl files")
    ap.add_argument("--glob", help="glob pattern (quote it)")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    files = list(args.files)
    if args.glob:
        files += globmod.glob(args.glob)
    files = [f for f in files if os.path.isfile(f) and not f.endswith(".bak")]
    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(1)

    tot_kept = tot_drop = 0
    for path in sorted(files):
        kept, dropped = clean_file(path, backup=not args.no_backup)
        tot_kept += kept
        tot_drop += dropped
        flag = "  <-- cleaned" if dropped else ""
        print(f"{os.path.basename(path)}: kept={kept} dropped={dropped}{flag}")
    print(f"\nTotal: kept={tot_kept} dropped={tot_drop} across {len(files)} files")


if __name__ == "__main__":
    main()
