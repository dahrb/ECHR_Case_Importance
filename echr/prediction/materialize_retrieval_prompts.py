"""Materialize model-independent retrieval prompts in parallel shards.

This stage performs every deterministic CPU operation before LLM inference:
candidate resolution, summary lookup, optional LegalBERT reranking, and final
prompt rendering. The resulting JSONL shards can be reused by any compatible
chat model and are consumed with ``run_retrieval_predictions --prompt_dir``.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from echr.prediction.run_retrieval_predictions import (
    DATA,
    build_appno_index,
    build_file_index,
    build_summary_index,
    get_examples,
    load_reranker,
    load_retrieval_results,
    render_retrieval_prompt,
)
from echr.prediction.prompts import ITER_LEVELS, iterative_retrieval_prompt


def default_output_dir(args, output_k=None) -> Path:
    k = args.k if output_k is None else output_k
    condition = f"{args.retriever}_k{k}"
    if args.rerank:
        condition += f"_rerank-{Path(args.rerank_model).name}"
    condition += (
        f"_text{args.text}_{args.split}_ctx{args.max_model_len}_out{args.max_tokens}"
    )
    prompt_root = "iterative_retrieval_v1" if args.iterative else "retrieval_v1"
    return Path(DATA) / "prompts" / prompt_root / f"article{args.article}" / condition


def render_iterative_prompts(row, article, examples, text, max_tokens, max_model_len):
    """Render all four iterative prompts within the retrieval context budget."""
    available_tokens = max(1000, max_model_len - max_tokens - 768)
    max_prompt_chars = int(available_tokens * 2.5)
    component_count = len(examples) + 1
    max_chars = max(500, (max_prompt_chars - 2500) // component_count)

    def render(limit):
        return {
            level: iterative_retrieval_prompt(
                row, str(article), level, examples, text, max_chars=limit,
            )
            for level in ITER_LEVELS
        }

    prompts = render(max_chars)
    while max(map(len, prompts.values())) > max_prompt_chars and max_chars > 500:
        shrink = max_prompt_chars / max(map(len, prompts.values()))
        next_max_chars = max(500, int(max_chars * shrink * 0.95))
        if next_max_chars >= max_chars:
            next_max_chars = max_chars - 1
        max_chars = next_max_chars
        prompts = render(max_chars)
    return prompts


def render_record(args, case_index, row, examples, output_k=None):
    k = args.k if output_k is None else output_k
    record = {
        "schema_version": 1,
        "article": str(args.article),
        "retriever": args.retriever,
        "k": k,
        "rerank": args.rerank,
        "reranker_model": os.path.abspath(args.rerank_model) if args.rerank else None,
        "text": args.text,
        "split": args.split,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "case_index": case_index,
        "Filename": row.get("Filename", f"row_{case_index}"),
        "importance": int(row["importance"]) if pd.notna(row.get("importance")) else None,
        "n_examples": len(examples),
        "example_filenames": list(examples),
    }
    if args.iterative:
        record["prompt_style"] = "iterative"
        record["level_prompts"] = render_iterative_prompts(
            row, args.article, examples, args.text, args.max_tokens,
            args.max_model_len,
        )
    else:
        record["prompt"] = render_retrieval_prompt(
            row,
            str(args.article),
            examples,
            args.text,
            args.max_tokens,
            args.max_model_len,
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, choices=["3", "6", "8"])
    parser.add_argument("--retriever", required=True,
                        choices=["bm25", "faiss", "ensemble_rrf", "gold", "tgn_kg"])
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--text", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank_model")
    parser.add_argument("--rerank_pool", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--case_batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--max_model_len", type=int, default=9600)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--iterative", action="store_true",
                        help="Emit four level-specific iterative prompts per case")
    parser.add_argument(
        "--all_k",
        action="store_true",
        help="For shared FAISS/BM25 reranking, score once and emit k=3,5,10",
    )
    args = parser.parse_args()

    if args.rerank and not args.rerank_model:
        raise ValueError("--rerank requires --rerank_model")
    if args.all_k and (not args.rerank or args.retriever not in {"faiss", "bm25"}):
        raise ValueError("--all_k is only valid for reranked FAISS or BM25")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    split_file = "comm_test.pkl" if args.split == "test" else "comm_valid.pkl"
    cases = pd.read_pickle(
        Path(DATA) / "processed" / f"article{args.article}" / "splits" / split_file
    )
    outcome_cases = pd.read_pickle(
        Path(DATA) / "processed" / f"article{args.article}" / "outcome_cases.pkl"
    )
    summaries = pd.read_pickle(
        Path(DATA) / "processed" / f"article{args.article}" / "outcome_summaries.pkl"
    )
    retrieval_results = load_retrieval_results(args.article, args.retriever, args.k)
    appno_index = build_appno_index(outcome_cases)
    file_index = build_file_index(outcome_cases)
    summary_index = build_summary_index(summaries)
    reranker = load_reranker(args.rerank_model) if args.rerank else None

    selected = [
        (case_index, row)
        for case_index, (_, row) in enumerate(cases.iterrows())
        if case_index % args.num_shards == args.shard_index
    ]
    output_ks = (3, 5, 10) if args.all_k else (args.k,)
    records_by_k = {k: [] for k in output_ks}
    for batch_start in range(0, len(selected), args.case_batch_size):
        batch = selected[batch_start:batch_start + args.case_batch_size]
        prepared = []
        pairs = []
        spans = []
        for case_index, row in batch:
            filename = row.get("Filename", f"row_{case_index}")
            candidates = get_examples(
                retrieval_results.get(filename, []),
                appno_index,
                summary_index,
                args.k,
                args.rerank_pool if reranker else None,
                file_index=file_index,
            )
            prepared.append((case_index, row, candidates))
            if reranker:
                start = len(pairs)
                query = str(row.get("Subject Matter", "")).lower()
                pairs.extend(
                    (
                        query,
                        (summary_index[file_id].get("200") or summary_500).lower(),
                    )
                    for file_id, (_, summary_500) in candidates.items()
                )
                spans.append((start, len(pairs)))

        scores = None
        if reranker and pairs:
            scores = reranker.predict(
                pairs,
                batch_size=args.batch_size,
                show_progress_bar=False,
            )

        for position, (case_index, row, candidates) in enumerate(prepared):
            ranked_file_ids = list(candidates)
            if reranker and candidates:
                start, end = spans[position]
                file_ids = list(candidates)
                ranked = sorted(zip(scores[start:end], file_ids), reverse=True)
                ranked_file_ids = [file_id for _, file_id in ranked]
            for output_k in output_ks:
                examples = {
                    file_id: candidates[file_id]
                    for file_id in ranked_file_ids[:output_k]
                }
                records_by_k[output_k].append(
                    render_record(args, case_index, row, examples, output_k)
                )

        print(
            f"[shard {args.shard_index}/{args.num_shards}] "
            f"prepared {min(batch_start + len(batch), len(selected))}/{len(selected)}",
            flush=True,
        )

    for output_k, records in records_by_k.items():
        output_dir = (
            Path(args.output_dir) / f"k{output_k}"
            if args.output_dir and args.all_k
            else Path(args.output_dir)
            if args.output_dir
            else default_output_dir(args, output_k)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"part-{args.shard_index:04d}-of-{args.num_shards:04d}.jsonl"
        )
        temporary_path = output_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        os.replace(temporary_path, output_path)
        print(f"Wrote {len(records)} prompts to {output_path}", flush=True)


if __name__ == "__main__":
    main()
