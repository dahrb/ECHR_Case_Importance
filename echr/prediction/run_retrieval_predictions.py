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
import glob
import json
import re
import os
import re
import sys
import time

# Per-session unique prefix: forces a prefix-cache miss on the vLLM server so
# reasoning_effort=low is applied on fresh KV state (not reused from a prior
# request that was cached without it).  os.urandom ensures different values
# across concurrent SLURM jobs even if they start at the same second.
_CACHE_BUSTER = os.urandom(4).hex() + " "

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import retrieval_prompt

# Optional re-ranker — only imported when --rerank is set
_reranker = None


def load_reranker(model_path: str):
    global _reranker
    if _reranker is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "rerank"))
        # Slurm launches many independent 4-CPU shards.  PyTorch otherwise
        # defaults its inter-op pool to the full host, causing severe
        # oversubscription when several shards share a node.
        import torch

        threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Another caller may already have started Torch parallel work.
            pass
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


def load_retrieval_results(article: str, retriever: str, k: int = None) -> dict:
    vdb = os.path.join(DATA, "vectordb", f"article{article}")
    if retriever == "bm25":
        path = os.path.join(vdb, "bm25_results.pkl")
    elif retriever == "ensemble_rrf":
        path = os.path.join(vdb, "ensemble_rrf_results.pkl")
    elif retriever == "gold":
        path = os.path.join(vdb, "gold_results.pkl")
    elif retriever == "tgn_kg":
        if k is None:
            raise ValueError("TGN KG retrieval requires k because its seed set depends on k")
        path = os.path.join(vdb, f"tgn_kg_k{k}_results.pkl")
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


def build_file_index(outcome_cases: pd.DataFrame) -> dict:
    """Build exact outcome-file index for temporally resolved KG results."""
    return {row["File"]: row for _, row in outcome_cases.iterrows()}


def build_summary_index(summaries: pd.DataFrame) -> dict:
    """Build constant-time summary lookup keyed by outcome filename."""
    return {
        row["Filename"]: {
            "200": str(row.get("200 Word Summary", "") or ""),
            "500": str(row.get("500 Word Summary", "") or ""),
        }
        for _, row in summaries.iterrows()
    }


def get_examples(
    retrieved_appnos: list,
    appno_index: dict,
    summaries: dict,
    k: int,
    n_retrieve: int = None,
    file_index: dict = None,
    query_date=None,
) -> dict:
    """
    Map top retrieved appnos to {file_id: (importance, summary)} dict.
    Returns up to n_retrieve entries (use k when not reranking).
    query_date: if provided, only outcome documents strictly before this date are used.
    """
    limit = n_retrieve if n_retrieve is not None else k
    examples = {}
    for appno_raw in retrieved_appnos:
        if len(examples) >= limit:
            break
        if file_index and appno_raw in file_index:
            # KG results identify the exact pre-query graph node. Do not remap
            # it to a later outcome document for the same application.
            row = file_index[appno_raw]
        else:
            first = appno_raw.split(";")[0].strip()
            rows = appno_index.get(first, [])
            if not rows:
                continue
            # Enforce query date cutoff: exclude docs on or after outcome date.
            if query_date is not None:
                rows = [r for r in rows if r["date"] < query_date]
            if not rows:
                continue
            # VDB/BM25 results are application numbers; preserve legacy mapping.
            row = sorted(rows, key=lambda r: r["date"])[-1]
        file_id = row["File"]
        importance = int(row["importance"])
        summary_record = summaries.get(file_id)
        if not summary_record:
            continue
        summary = summary_record.get("500", "")
        if not summary or len(str(summary)) < 50:
            continue
        examples[file_id] = (importance, str(summary))
    return examples


def rerank_examples(
    query_text: str,
    examples: dict,
    k: int,
    reranker,
    rerank_texts: dict = None,
) -> dict:
    """Re-rank with the training-matched 200-word candidate summaries.

    The returned ``examples`` retain 500-word summaries for the LLM prompt.
    This deliberately separates retrieval scoring text from demonstration text.
    """
    if len(examples) <= k:
        return examples
    file_ids = list(examples.keys())
    rerank_texts = rerank_texts or {}
    pairs = [
        (query_text.lower(), rerank_texts.get(fid, examples[fid][1]).lower())
        for fid in file_ids
    ]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, file_ids), reverse=True)
    return {fid: examples[fid] for _, fid in ranked[:k]}


def render_retrieval_prompt(
    row,
    article: str,
    examples: dict,
    text: int,
    max_tokens: int,
    max_model_len: int,
) -> str:
    """Render a retrieval prompt within a conservative aggregate context budget."""
    available_tokens = max(1000, max_model_len - max_tokens - 768)
    max_prompt_chars = int(available_tokens * 2.5)
    component_count = len(examples) + 1
    max_chars = max(500, (max_prompt_chars - 2500) // component_count)
    prompt = retrieval_prompt(row, article, examples, text=text, max_chars=max_chars)
    while len(prompt) > max_prompt_chars and max_chars > 500:
        shrink = max_prompt_chars / len(prompt)
        next_max_chars = max(500, int(max_chars * shrink * 0.95))
        if next_max_chars >= max_chars:
            next_max_chars = max_chars - 1
        max_chars = next_max_chars
        prompt = retrieval_prompt(row, article, examples, text=text, max_chars=max_chars)
    return prompt


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
    prompt_override: str = None,
    n_examples_override: int = None,
) -> dict:
    # Keep the complete rendered prompt comfortably below the server context
    # window. A per-field minimum is unsafe for k=10 because ten independently
    # truncated demonstrations can exceed the intended aggregate budget.
    prompt = prompt_override or render_retrieval_prompt(
        row, article, examples, text, max_tokens, max_model_len
    )
    n_examples = len(examples) if n_examples_override is None else n_examples_override
    extra = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
        # Prefix-cache isolation: concurrent requests share the same system-message
        # tokens; a stale or wrong-effort cache entry would silently suppress RE.
        # cache_salt makes each request's cache key unique, ensuring fresh KV
        # computation with the correct reasoning_effort on every call.
        extra["cache_salt"] = os.urandom(8).hex()
    if no_thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
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
            if pred is None and attempt < max_retries - 1:
                print(
                    f"  Retry {attempt + 1}/{max_retries}: invalid importance "
                    f"label {pred_raw!r}",
                    flush=True,
                )
                messages.extend([
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Your Case Importance value must be exactly one of: "
                            "1, 2, 3, key_case. Reassess the case and return only "
                            "the required JSON object with one of those values."
                        ),
                    },
                ])
                continue
            return {
                "raw_prediction": pred_raw,
                "prediction": pred,
                "reasoning": parsed.get("Reasoning", ""),
                "raw_output": raw,
                "n_examples": n_examples,
            }
        except (json.JSONDecodeError, KeyError) as e:
            match = re.search(r'"Case Importance"\s*:\s*"(key_case|[1-3])"', raw)
            if match:
                pred_raw = match.group(1).lower()
                return {"raw_prediction": pred_raw, "prediction": IMPORTANCE_MAP[pred_raw],
                        "reasoning": "", "raw_output": raw, "n_examples": n_examples}
            if attempt < max_retries - 1:
                print(
                    f"  Retry {attempt + 1}/{max_retries}: invalid JSON: {e}",
                    flush=True,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Return only valid JSON with Case Importance set to exactly "
                        "one of: 1, 2, 3, key_case."
                    ),
                })
                continue
            return {
                "raw_prediction": raw if "raw" in dir() else "",
                "prediction": None,
                "reasoning": "",
                "raw_output": raw if "raw" in dir() else "",
                "n_examples": n_examples,
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
    parser.add_argument("--prompt_dir", default=None,
                        help="Read complete pre-materialized prompt shards from this directory")
    parser.add_argument("--run_tag", default=None,
                        help="Append a tag to the result filename to isolate a rerun from legacy output")
    args = parser.parse_args()

    if args.rerank and not args.rerank_model and not args.prompt_dir:
        raise ValueError("--rerank requires --rerank_model <path>")

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(f"No endpoint and {ENDPOINT_FILE} not found.")

    reranker = load_reranker(args.rerank_model) if args.rerank and not args.prompt_dir else None

    print(
        f"[art {args.article}] vLLM={endpoint} model={args.model} "
        f"retriever={args.retriever} k={args.k} rerank={args.rerank} "
        f"rerank_candidate_summary={'200-word' if args.rerank else 'n/a'} split={args.split}",
        flush=True,
    )
    client = get_client(endpoint)

    prompt_records = None
    if args.prompt_dir:
        prompt_files = sorted(glob.glob(os.path.join(args.prompt_dir, "part-*-of-*.jsonl")))
        shard_pattern = re.compile(r"part-(\d+)-of-(\d+)\.jsonl$")
        shard_pairs = [shard_pattern.search(path) for path in prompt_files]
        if not prompt_files or any(match is None for match in shard_pairs):
            raise RuntimeError(f"No valid prompt shards found in {args.prompt_dir}")
        shard_totals = {int(match.group(2)) for match in shard_pairs}
        shard_indices = {int(match.group(1)) for match in shard_pairs}
        if len(shard_totals) != 1:
            raise RuntimeError(f"Inconsistent prompt shard totals in {args.prompt_dir}")
        shard_total = shard_totals.pop()
        if shard_indices != set(range(shard_total)):
            missing = sorted(set(range(shard_total)) - shard_indices)
            raise RuntimeError(f"Incomplete prompt directory {args.prompt_dir}; missing shards {missing}")

        by_filename = {}
        for path in prompt_files:
            with open(path) as handle:
                for line in handle:
                    record = json.loads(line)
                    if str(record["article"]) != str(args.article):
                        raise ValueError(f"Article mismatch in {path}: {record['article']}")
                    if record["retriever"] != args.retriever or int(record["k"]) != args.k:
                        raise ValueError(f"Retrieval configuration mismatch in {path}")
                    if bool(record["rerank"]) != args.rerank:
                        raise ValueError(f"Rerank configuration mismatch in {path}")
                    if int(record["text"]) != args.text or record["split"] != args.split:
                        raise ValueError(f"Prompt configuration mismatch in {path}")
                    if int(record["max_model_len"]) > args.max_model_len:
                        raise ValueError(
                            f"Prompt context budget exceeds --max_model_len in {path}"
                        )
                    if int(record["max_tokens"]) < args.max_tokens:
                        raise ValueError(
                            f"Prompt output reserve is below --max_tokens in {path}"
                        )
                    if record["Filename"] in by_filename:
                        raise ValueError(f"Duplicate prompt for {record['Filename']} in {path}")
                    by_filename[record["Filename"]] = record
        prompt_records = sorted(by_filename.values(), key=lambda record: record["case_index"])
        case_indices = [int(record["case_index"]) for record in prompt_records]
        if case_indices != list(range(len(prompt_records))):
            raise RuntimeError(
                f"Prompt directory {args.prompt_dir} has missing or duplicate case indices"
            )
        cases = prompt_records
        print(
            f"[art {args.article}] loaded {len(cases)} materialized prompts "
            f"from {shard_total} shards",
            flush=True,
        )
    else:
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
        summary_index = build_summary_index(summaries)
        retrieval_results = load_retrieval_results(args.article, args.retriever, args.k)
        appno_index = build_appno_index(outcome_cases)
        file_index = build_file_index(outcome_cases)
        print(
            f"[art {args.article}] outcome_cases={len(outcome_cases)}, "
            f"summaries={len(summaries)}, retrieval_results={len(retrieval_results)}",
            flush=True,
        )

    safe_model = args.model.replace("/", "_").replace(":", "_")
    rr_suffix = "_rerank" if args.rerank else ""
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    out_name = (
        f"retrieval_{args.retriever}_k{args.k}{rr_suffix}_text{args.text}_{args.split}_{safe_model}{tag_suffix}.jsonl"
    )
    out_dir = os.path.join(DATA, "results", f"article{args.article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    _MODEL_ALIASES = {"Llama-3.3-70B": "Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-Instruct-FP8": "Llama-3.3-70B"}
    done_filenames: set = set()
    if args.resume:
        _alias = _MODEL_ALIASES.get(safe_model) if not args.run_tag else None
        _paths = [out_path] + ([os.path.join(out_dir, out_name.replace(safe_model, _alias))] if _alias else [])
        for _p in _paths:
            if os.path.exists(_p):
                with open(_p) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if record.get("prediction") is not None:
                                done_filenames.add(record["Filename"])
                        except Exception:
                            pass
        if done_filenames:
            print(f"  Resuming: {len(done_filenames)} already done", flush=True)

    if args.limit:
        cases = cases[:args.limit] if prompt_records is not None else cases.head(args.limit)

    with open(out_path, "a" if args.resume else "w") as fout:
        rows = enumerate(cases) if prompt_records is not None else cases.iterrows()
        for i, row in rows:
            filename = row.get("Filename", f"row_{i}")
            if filename in done_filenames:
                continue

            if i % 100 == 0:
                print(f"  [{i}/{len(cases)}] {filename}", flush=True)

            if prompt_records is not None:
                examples = {}
                result = predict_one(
                    client, row, args.article, args.model, examples, text=args.text,
                    max_tokens=args.max_tokens, max_model_len=args.max_model_len,
                    reasoning_effort=args.reasoning_effort,
                    no_thinking=args.no_thinking,
                    prompt_override=row["prompt"],
                    n_examples_override=int(row["n_examples"]),
                )
            else:
                retrieved = retrieval_results.get(filename, [])
                n_retrieve = args.rerank_pool if reranker else None
                _doc_date = row.get("doc_date")
                _query_date = pd.Timestamp(_doc_date) if _doc_date else None
                examples = get_examples(
                    retrieved,
                    appno_index,
                    summary_index,
                    args.k,
                    n_retrieve,
                    file_index=file_index,
                    query_date=_query_date,
                )
                if reranker and examples:
                    query_text = str(row.get("Subject Matter", ""))
                    rerank_texts = {
                        file_id: summary_index[file_id].get("200") or summary_500
                        for file_id, (_, summary_500) in examples.items()
                    }
                    examples = rerank_examples(
                        query_text, examples, args.k, reranker, rerank_texts
                    )

                result = predict_one(
                    client, row, args.article, args.model, examples, text=args.text,
                    max_tokens=args.max_tokens, max_model_len=args.max_model_len,
                    reasoning_effort=args.reasoning_effort,
                    no_thinking=args.no_thinking,
                )
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
