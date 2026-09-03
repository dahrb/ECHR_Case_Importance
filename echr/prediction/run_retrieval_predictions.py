"""
Run retrieval-augmented (RAG) importance predictions using BM25 or FAISS top-K.

For each comm-phase test case:
  1. Look up top-K pre-computed retrieval results (appno list).
  2. Map appnos to outcome_cases rows to get File IDs + importance labels.
  3. Fetch 500-word summaries from outcome_summaries.pkl.
  4. Build a retrieval-augmented prompt and call the vLLM API.

Usage:
    python echr/prediction/run_retrieval_predictions.py --article 3 --retriever bm25 --k 3
    python echr/prediction/run_retrieval_predictions.py --article 6 --retriever faiss --k 5

Output:
    data/results/article{N}/retrieval_{retriever}_k{K}_text{T}_{split}_{model}.jsonl
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import retrieval_prompt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_endpoint.txt")

IMPORTANCE_MAP = {"key_case": 1, "1": 2, "2": 3, "3": 4}


def get_client(endpoint: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=endpoint)


def load_retrieval_results(article: str, retriever: str) -> dict:
    vdb = os.path.join(DATA, "vectordb", f"article{article}")
    if retriever == "bm25":
        path = os.path.join(vdb, "bm25_results.pkl")
    elif retriever == "ensemble_rrf":
        path = os.path.join(vdb, "ensemble_rrf_results.pkl")
    elif retriever == "gold":
        path = os.path.join(vdb, "gold_results.pkl")
    elif retriever == "tgn_kg":
        path = os.path.join(vdb, "tgn_kg_results.pkl")
    else:  # faiss (default)
        path = os.path.join(vdb, "cosine_qwen3-8b_raw_chunk_2048_results.pkl")
    return pd.read_pickle(path)


def build_appno_index(outcome_cases: pd.DataFrame) -> dict:
    """Build index: first_appno_part → list of rows, for fast lookup."""
    idx = {}
    for _, row in outcome_cases.iterrows():
        key = str(row["appno"]).split(";")[0].strip()
        idx.setdefault(key, []).append(row)
    return idx


def get_examples(
    retrieved_appnos: list,
    appno_index: dict,
    summaries: pd.DataFrame,
    k: int,
) -> dict:
    """
    Map top retrieved appnos to {file_id: (importance, summary)} dict, up to k entries.
    """
    examples = {}
    for appno_raw in retrieved_appnos:
        if len(examples) >= k:
            break
        first = appno_raw.split(";")[0].strip()
        rows = appno_index.get(first, [])
        if not rows:
            continue
        # take the most recent case
        row = sorted(rows, key=lambda r: r["date"])[-1]
        file_id = row["File"]
        importance = int(row["importance"])
        summ_rows = summaries[summaries["Filename"] == file_id]
        if summ_rows.empty:
            continue
        summary = summ_rows.iloc[0].get("500 Word Summary", "")
        if not summary or len(str(summary)) < 50:
            continue
        examples[file_id] = (importance, str(summary))
    return examples


def predict_one(
    client: OpenAI,
    row,
    article: str,
    model: str,
    examples: dict,
    text: int = 1,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    prompt = retrieval_prompt(row, article, examples, text=text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.0,
                seed=42,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            if not raw.startswith("{"):
                match = re.search(r"\{", raw)
                if match:
                    raw = raw[match.start():]
            parsed = json.loads(raw)
            pred_raw = str(parsed.get("Case Importance", "")).strip().lower()
            pred = IMPORTANCE_MAP.get(pred_raw, None)
            return {
                "raw_prediction": pred_raw,
                "prediction": pred,
                "reasoning": parsed.get("Reasoning", ""),
                "raw_output": raw,
                "n_examples": len(examples),
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "raw_prediction": raw if "raw" in dir() else "",
                "prediction": None,
                "reasoning": "",
                "raw_output": raw if "raw" in dir() else "",
                "n_examples": len(examples),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"  FAILED: {e}", flush=True)
                return {
                    "raw_prediction": "", "prediction": None,
                    "reasoning": "", "raw_output": "", "n_examples": 0,
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="3, 6, or 8")
    parser.add_argument("--model", default="gpt-oss-20b")
    parser.add_argument("--retriever", default="bm25",
                        choices=["bm25", "faiss", "ensemble_rrf", "gold", "tgn_kg"])
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved examples")
    parser.add_argument("--text", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--split", default="test", choices=["test", "valid"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(f"No endpoint and {ENDPOINT_FILE} not found.")

    print(
        f"[art {args.article}] vLLM={endpoint} model={args.model} "
        f"retriever={args.retriever} k={args.k} split={args.split}",
        flush=True,
    )
    client = get_client(endpoint)

    split_file = "comm_test.pkl" if args.split == "test" else "comm_valid.pkl"
    cases_path = os.path.join(
        DATA, "processed", f"article{args.article}", "splits", split_file
    )
    cases = pd.read_pickle(cases_path)
    print(f"[art {args.article}] {len(cases)} cases in {args.split} split", flush=True)

    outcome_cases = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{args.article}", "outcome_cases.pkl")
    )
    summaries = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{args.article}", "outcome_summaries.pkl")
    )
    retrieval_results = load_retrieval_results(args.article, args.retriever)
    appno_index = build_appno_index(outcome_cases)
    print(
        f"[art {args.article}] outcome_cases={len(outcome_cases)}, "
        f"summaries={len(summaries)}, retrieval_results={len(retrieval_results)}",
        flush=True,
    )

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_name = (
        f"retrieval_{args.retriever}_k{args.k}_text{args.text}_{args.split}_{safe_model}.jsonl"
    )
    out_dir = os.path.join(DATA, "results", f"article{args.article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    done_filenames: set = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_filenames.add(json.loads(line)["Filename"])
                except Exception:
                    pass
        print(f"  Resuming: {len(done_filenames)} already done", flush=True)

    if args.limit:
        cases = cases.head(args.limit)

    with open(out_path, "a" if args.resume else "w") as fout:
        for i, (_, row) in enumerate(cases.iterrows()):
            filename = row.get("Filename", f"row_{i}")
            if filename in done_filenames:
                continue

            if i % 100 == 0:
                print(f"  [{i}/{len(cases)}] {filename}", flush=True)

            retrieved = retrieval_results.get(filename, [])
            examples = get_examples(retrieved, appno_index, summaries, args.k)

            result = predict_one(client, row, args.article, args.model, examples, text=args.text)
            record = {
                "Filename": filename,
                "importance": int(row["importance"]) if pd.notna(row.get("importance")) else None,
                **result,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[art {args.article}] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
