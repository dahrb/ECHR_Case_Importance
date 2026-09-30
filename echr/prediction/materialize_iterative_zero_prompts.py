"""Materialize four level-specific iterative zero-shot prompts in CPU shards."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from echr.prediction.prompts import ITER_LEVELS, iterative_prompt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, choices=["3", "6", "8"])
    parser.add_argument("--text", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--max_model_len", type=int, default=48000)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    split_file = "comm_test.pkl" if args.split == "test" else "comm_valid.pkl"
    cases = pd.read_pickle(DATA / "processed" / f"article{args.article}" / "splits" / split_file)
    records = []
    # Match the retrieval materializer's conservative character budget: reserve
    # generation and template overhead, then keep every level prompt inside the
    # requested server context window.
    available_tokens = max(1000, args.max_model_len - args.max_tokens - 768)
    max_prompt_chars = int(available_tokens * 2.5)
    max_case_chars = max(500, max_prompt_chars - 2500)
    for case_index, (_, row) in enumerate(cases.iterrows()):
        if case_index % args.num_shards != args.shard_index:
            continue
        records.append({
            "schema_version": 1,
            "prompt_style": "iterative",
            "article": str(args.article),
            "retriever": "base",
            "k": 0,
            "rerank": False,
            "reranker_model": None,
            "text": args.text,
            "split": args.split,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "case_index": case_index,
            "Filename": row.get("Filename", f"row_{case_index}"),
            "importance": int(row["importance"]) if pd.notna(row.get("importance")) else None,
            "n_examples": 0,
            "example_filenames": [],
            "level_prompts": {
                level: iterative_prompt(
                    row, str(args.article), level, args.text,
                    max_chars=max_case_chars,
                )
                for level in ITER_LEVELS
            },
        })

    output_dir = (
        DATA / "prompts" / "iterative_zero_v1" / f"article{args.article}"
        / f"base_text{args.text}_{args.split}_ctx{args.max_model_len}_out{args.max_tokens}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"part-{args.shard_index:04d}-of-{args.num_shards:04d}.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(temporary_path, output_path)
    print(f"Wrote {len(records)} prompts to {output_path}")


if __name__ == "__main__":
    main()
