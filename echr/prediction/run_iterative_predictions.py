"""
Iterative-Prompting predictions (Exp 2, §5.3).

For each test case, issue one LLM call per importance level (key_case, 1, 2, 3),
asking whether the case matches that level and for a confidence score.
The final prediction is the level with the highest confidence on a "Yes" answer
(or the highest raw confidence if no "Yes" is returned).

Output:
    data/results/article{N}/iterative_{split}_{model}.jsonl
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import iterative_prompt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_endpoint.txt")

LEVELS = ["key_case", "1", "2", "3"]
LEVEL_TO_INT = {"key_case": 1, "1": 2, "2": 3, "3": 4}
IMPORTANCE_MAP = {"key_case": 1, "1": 2, "2": 3, "3": 4}


def get_client(endpoint: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=endpoint)


def query_level(client, row, article, model, level_key, text, prompt=None,
                max_tokens=600, reasoning_effort="low", max_retries=12, retry_delay=5.0):
    """Call the LLM for one level. Returns (matches_yes: bool, confidence: float, reasoning: str)."""
    prompt = prompt or iterative_prompt(row, article, level_key, text=text)
    for attempt in range(max_retries):
        try:
            # A few long-context cases spend most of a small completion budget
            # on hidden reasoning and return ``content=None`` with
            # ``finish_reason=length``.  Escalate only retry attempts, capped
            # at 3k, so normal iterative calls retain the fast 1k budget.
            attempt_max_tokens = min(max_tokens * (2 ** attempt), 3000)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=attempt_max_tokens,
                temperature=0.0,
                seed=42,
                extra_body={"chat_template_kwargs": {"reasoning_effort": reasoning_effort}},
            )
            _content = resp.choices[0].message.content
            if _content is None:
                raise ValueError("API returned None content")
            raw = _content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            if not raw.startswith("{"):
                m = re.search(r"\{", raw)
                if m:
                    raw = raw[m.start():]
            parsed = json.loads(raw)
            matches = str(parsed.get("Matches", "")).strip().lower() == "yes"
            try:
                confidence = float(parsed.get("Confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("Reasoning", ""))
            return matches, confidence, reasoning, raw
        except (json.JSONDecodeError, KeyError) as e:
            # Empty/non-JSON generations are transient under a busy vLLM
            # engine.  Retry them; silently turning them into a 0-confidence
            # level biases the iterative argmax.
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False, 0.0, str(e), raw if "raw" in dir() else ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False, 0.0, str(e), ""
    return False, 0.0, "", ""


def predict_iterative(client, row, article, model, text, level_prompts=None,
                      max_tokens=600, reasoning_effort="low", level_concurrency=4):
    """Run all 4 level queries and return the predicted level."""
    # The four independent binary decisions are submitted together.  This
    # preserves the experimental decision rule while feeding vLLM a useful
    # batch and lets cache-friendly prompt files share their common prefix.
    def run_one(level_key):
        return level_key, query_level(
            client, row, article, model, level_key, text,
            prompt=(level_prompts or {}).get(level_key), max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    level_results = {}
    with ThreadPoolExecutor(max_workers=min(level_concurrency, len(LEVELS))) as pool:
        for level_key, (matches, confidence, reasoning, raw) in pool.map(run_one, LEVELS):
            level_results[level_key] = {
                "matches": matches,
                "confidence": confidence,
                "reasoning": reasoning,
                "raw": raw,
            }

    # Pick level with highest confidence among "Yes" answers
    yes_levels = {k: v["confidence"] for k, v in level_results.items() if v["matches"]}
    if yes_levels:
        best_level = max(yes_levels, key=yes_levels.get)
    else:
        # Fall back to raw confidence argmax
        best_level = max(level_results, key=lambda k: level_results[k]["confidence"])

    pred_int = LEVEL_TO_INT.get(best_level)
    return {
        "prediction": pred_int,
        "predicted_level": best_level,
        "level_results": level_results,
    }


def load_prompt_records(prompt_dir: str):
    """Read CPU-materialized iterative prompts, retaining deterministic case order."""
    records = []
    seen = set()
    for path in sorted(Path(prompt_dir).glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                filename = record["Filename"]
                if filename in seen:
                    raise ValueError(f"Duplicate Filename in prompt directory: {filename}")
                if set(record.get("level_prompts", {})) != set(LEVELS):
                    raise ValueError(f"Incomplete level prompts for {filename}")
                seen.add(filename)
                records.append(record)
    if not records:
        raise ValueError(f"No materialized prompt JSONL files found in {prompt_dir}")
    return sorted(records, key=lambda record: record.get("case_index", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True)
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--text", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-dir", default=None,
                        help="Directory of CPU-materialized iterative JSONL prompt shards")
    parser.add_argument("--run-tag", default=None,
                        help="Suffix used to keep condition outputs separate")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--level-concurrency", type=int, default=4,
                        help="Concurrent independent level calls per case")
    args = parser.parse_args()

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(f"No endpoint and {ENDPOINT_FILE} not found.")

    print(f"[art {args.article}] vLLM={endpoint} model={args.model} split={args.split} text={args.text}", flush=True)
    client = get_client(endpoint)

    if args.prompt_dir:
        cases = load_prompt_records(args.prompt_dir)
    else:
        split_file = {"test": "comm_test.pkl", "valid": "comm_valid.pkl", "train": "comm_train.pkl"}[args.split]
        cases = pd.read_pickle(os.path.join(DATA, "processed", f"article{args.article}", "splits", split_file))
    print(f"[art {args.article}] {len(cases)} cases in {args.split} split", flush=True)

    if args.limit:
        cases = cases[:args.limit] if args.prompt_dir else cases.head(args.limit)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_name = f"iterative_text{args.text}_{args.split}{tag}_{safe_model}.jsonl"
    out_dir = os.path.join(DATA, "results", f"article{args.article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    _MODEL_ALIASES = {"Llama-3.3-70B": "Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-Instruct-FP8": "Llama-3.3-70B"}
    done: set = set()
    if args.resume:
        _alias = _MODEL_ALIASES.get(safe_model)
        _paths = [out_path] + ([os.path.join(out_dir, out_name.replace(safe_model, _alias))] if _alias else [])
        for _p in _paths:
            if os.path.exists(_p):
                with open(_p) as f:
                    for line in f:
                        try:
                            done.add(json.loads(line)["Filename"])
                        except Exception:
                            pass
        if done:
            print(f"  Resuming: {len(done)} already done", flush=True)

    with open(out_path, "a" if args.resume else "w") as fout:
        iterator = cases if args.prompt_dir else (row for _, row in cases.iterrows())
        for i, row in enumerate(iterator):
            filename = row.get("Filename", f"row_{i}")
            if filename in done:
                continue
            if i % 50 == 0:
                print(f"  [{i}/{len(cases)}] {filename}", flush=True)

            result = predict_iterative(
                client, row, args.article, args.model, args.text,
                level_prompts=row.get("level_prompts") if args.prompt_dir else None,
                max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
                level_concurrency=args.level_concurrency,
            )
            failed_levels = [
                level for level, value in result["level_results"].items()
                if not value.get("raw")
            ]
            if failed_levels:
                raise RuntimeError(
                    f"Unresolved iterative responses for {filename}: {failed_levels}; "
                    "refusing to write a biased fallback prediction"
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
