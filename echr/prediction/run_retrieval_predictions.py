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
import sys
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import retrieval_prompt

# Optional re-ranker — only imported when --rerank is set
_reranker = None


def load_reranker(model_path: str):
    global _reranker
    if _reranker is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "rerank"))
        from sentence_transformers.cross_encoder import CrossEncoder
        _reranker = CrossEncoder(model_path)
        print(f"[reranker] Loaded from {model_path}", flush=True)
    return _reranker

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
    n_retrieve: int = None,
) -> dict:
    """
    Map top retrieved appnos to {file_id: (importance, summary)} dict.
    Returns up to n_retrieve entries (use k when not reranking).
    """
    limit = n_retrieve if n_retrieve is not None else k
    examples = {}
    for appno_raw in retrieved_appnos:
        if len(examples) >= limit:
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


def rerank_examples(
    query_text: str,
    examples: dict,
    k: int,
    reranker,
) -> dict:
    """Re-rank candidate examples by CrossEncoder score, return top-k."""
    if len(examples) <= k:
        return examples
    file_ids = list(examples.keys())
    pairs = [(query_text.lower(), examples[fid][1].lower()) for fid in file_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, file_ids), reverse=True)
    return {fid: examples[fid] for _, fid in ranked[:k]}


def predict_one(
    client: OpenAI,
    row,
    article: str,
    model: str,
    examples: dict,
    text: int = 1,
    max_tokens: int = 1200,
    max_model_len: int = 32000,
    reasoning_effort: str = None,
    no_thinking: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    # Budget: model context - output tokens - prompt template overhead (~500 tok) - summary chars
    # Use 3 chars/token (conservative for legal text); ensure at least 5000 chars
    summaries_chars = sum(len(str(s)) for _, s in examples.values()) if examples else 0
    available_tokens = int((max_model_len - max_tokens - 500) * 0.90)  # 10% safety margin
    max_chars = max(5000, available_tokens * 3 - summaries_chars - 1500)
    prompt = retrieval_prompt(row, article, examples, text=text, max_chars=max_chars)
    extra = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    if no_thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body=extra,
                seed=42,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            raw = content.strip()
            # Extract JSON: handle code fences anywhere in response
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
            if m:
                raw = m.group(1)
            elif raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            elif not raw.startswith("{"):
                start = raw.find("{")
                if start != -1:
                    raw = raw[start:]
                    end = raw.rfind("```")
                    if end != -1:
                        raw = raw[:end]
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
    parser.add_argument("--rerank", action="store_true",
                        help="Apply LegalBERT cross-encoder reranking before selecting top-k")
    parser.add_argument("--rerank_model", default=None,
                        help="Path to saved CrossEncoder model dir (required if --rerank)")
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable thinking for LoRA FT models (RE=medium not enforced through --enable-lora)")
    parser.add_argument("--reasoning_effort", default=None, choices=["low","medium","high"],
                        help="gpt-oss reasoning effort; use low for FT model")
    parser.add_argument("--max_tokens", type=int, default=1200,
                        help="Generation budget (raise for reasoning models like gpt-oss FT)")
    parser.add_argument("--max_model_len", type=int, default=32000,
                        help="Server max_model_len; used to compute prompt budget")
    parser.add_argument("--rerank_pool", type=int, default=50,
                        help="Candidate pool size before reranking (default 50)")
    args = parser.parse_args()

    if args.rerank and not args.rerank_model:
        raise ValueError("--rerank requires --rerank_model <path>")

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(f"No endpoint and {ENDPOINT_FILE} not found.")

    reranker = load_reranker(args.rerank_model) if args.rerank else None

    print(
        f"[art {args.article}] vLLM={endpoint} model={args.model} "
        f"retriever={args.retriever} k={args.k} rerank={args.rerank} split={args.split}",
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
    rr_suffix = "_rerank" if args.rerank else ""
    out_name = (
        f"retrieval_{args.retriever}_k{args.k}{rr_suffix}_text{args.text}_{args.split}_{safe_model}.jsonl"
    )
    out_dir = os.path.join(DATA, "results", f"article{args.article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    _MODEL_ALIASES = {"Llama-3.3-70B": "Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-Instruct-FP8": "Llama-3.3-70B"}
    done_filenames: set = set()
    if args.resume:
        _alias = _MODEL_ALIASES.get(safe_model)
        _paths = [out_path] + ([os.path.join(out_dir, out_name.replace(safe_model, _alias))] if _alias else [])
        for _p in _paths:
            if os.path.exists(_p):
                with open(_p) as f:
                    for line in f:
                        try:
                            done_filenames.add(json.loads(line)["Filename"])
                        except Exception:
                            pass
        if done_filenames:
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
            n_retrieve = args.rerank_pool if reranker else None
            examples = get_examples(retrieved, appno_index, summaries, args.k, n_retrieve)
            if reranker and examples:
                query_text = str(row.get("Subject Matter", ""))
                examples = rerank_examples(query_text, examples, args.k, reranker)

            result = predict_one(client, row, args.article, args.model, examples, text=args.text,
                                 max_tokens=args.max_tokens, max_model_len=args.max_model_len,
                                 reasoning_effort=args.reasoning_effort,
                                 no_thinking=args.no_thinking)
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
