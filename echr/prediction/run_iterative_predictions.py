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
import json
import os
import re
import time

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


def query_level(client, row, article, model, level_key, text, max_retries=3, retry_delay=5.0):
    """Call the LLM for one level. Returns (matches_yes: bool, confidence: float, reasoning: str)."""
    prompt = iterative_prompt(row, article, level_key, text=text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
                seed=42,
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
        except (json.JSONDecodeError, KeyError):
            return False, 0.0, "", raw if "raw" in dir() else ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False, 0.0, str(e), ""
    return False, 0.0, "", ""


def predict_iterative(client, row, article, model, text):
    """Run all 4 level queries and return the predicted level."""
    level_results = {}
    for level_key in LEVELS:
        matches, confidence, reasoning, raw = query_level(
            client, row, article, model, level_key, text
        )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True)
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--text", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(f"No endpoint and {ENDPOINT_FILE} not found.")

    print(f"[art {args.article}] vLLM={endpoint} model={args.model} split={args.split} text={args.text}", flush=True)
    client = get_client(endpoint)

    split_file = {"test": "comm_test.pkl", "valid": "comm_valid.pkl", "train": "comm_train.pkl"}[args.split]
    cases = pd.read_pickle(os.path.join(DATA, "processed", f"article{args.article}", "splits", split_file))
    print(f"[art {args.article}] {len(cases)} cases in {args.split} split", flush=True)

    if args.limit:
        cases = cases.head(args.limit)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_name = f"iterative_text{args.text}_{args.split}_{safe_model}.jsonl"
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
        for i, (_, row) in enumerate(cases.iterrows()):
            filename = row.get("Filename", f"row_{i}")
            if filename in done:
                continue
            if i % 50 == 0:
                print(f"  [{i}/{len(cases)}] {filename}", flush=True)

            result = predict_iterative(client, row, args.article, args.model, args.text)
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
