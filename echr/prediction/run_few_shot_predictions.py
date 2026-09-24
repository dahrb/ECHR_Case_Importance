"""
Run few-shot importance predictions using one fixed example per importance class
drawn from comm_valid.pkl (random_state=42, one per class).

Usage:
    python echr/prediction/run_few_shot_predictions.py --article 3
    python echr/prediction/run_few_shot_predictions.py --article 6 --model gpt-oss-120b

Output:
    data/results/article{N}/few_shot_text{T}_{split}_{model}.jsonl
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import few_shot_prompt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_endpoint.txt")

IMPORTANCE_MAP = {"key_case": 1, "1": 2, "2": 3, "3": 4}


def get_client(endpoint: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=endpoint)


def build_fixed_examples(valid_cases: pd.DataFrame, text: int = 1) -> dict:
    """
    Pick one random case per importance class from valid split (random_state=42).
    Returns {importance_int: text_str, ...}
    """
    examples = {}
    match text:
        case 1:
            col = "Subject Matter"
        case 2:
            col = "Questions"
        case 3:
            col = None  # handled below
        case _:
            raise ValueError(f"Invalid text: {text}")

    for imp in sorted(valid_cases["importance"].dropna().unique()):
        imp = int(imp)
        pool = valid_cases[valid_cases["importance"] == imp]
        if pool.empty:
            continue
        row = pool.sample(1, random_state=42).iloc[0]
        if col is not None:
            examples[imp] = str(row.get(col, ""))
        else:
            examples[imp] = str(row.get("Subject Matter", "")) + " " + str(row.get("Questions", ""))
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
    # Truncation budgets (chars per field): full → 4000 → 1500 on context overflow
    truncation_budgets = [None, 4000, 1500]
    for attempt in range(max_retries):
        max_chars = truncation_budgets[min(attempt, len(truncation_budgets) - 1)]
        if max_chars is not None:
            print(f"  Retrying with max_chars={max_chars} per field", flush=True)
        prompt = few_shot_prompt(row, article, examples, text=text, max_chars=max_chars)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.0,
                seed=42,
            )
            _content = resp.choices[0].message.content
            if _content is None:
                raise ValueError("API returned None content")
            raw = _content.strip()
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
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "raw_prediction": raw if "raw" in dir() else "",
                "prediction": None,
                "reasoning": "",
                "raw_output": raw if "raw" in dir() else "",
            }
        except Exception as e:
            err_str = str(e)
            if "max_tokens must be at least 1" in err_str or "context_length_exceeded" in err_str:
                print(f"  Context overflow (attempt {attempt+1}): {e}", flush=True)
                continue  # immediately retry with smaller truncation_budget
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"  FAILED: {e}", flush=True)
                return {"raw_prediction": "", "prediction": None, "reasoning": "", "raw_output": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True)
    parser.add_argument("--model", default="gpt-oss-20b")
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
        f"split={args.split} text={args.text}",
        flush=True,
    )
    client = get_client(endpoint)

    valid_path = os.path.join(DATA, "processed", f"article{args.article}", "splits", "comm_valid.pkl")
    valid_cases = pd.read_pickle(valid_path)
    examples = build_fixed_examples(valid_cases, text=args.text)
    print(f"[art {args.article}] fixed examples: {list(examples.keys())}", flush=True)

    split_file = "comm_test.pkl" if args.split == "test" else "comm_valid.pkl"
    cases_path = os.path.join(DATA, "processed", f"article{args.article}", "splits", split_file)
    cases = pd.read_pickle(cases_path)
    print(f"[art {args.article}] {len(cases)} cases in {args.split} split", flush=True)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_name = f"few_shot_text{args.text}_{args.split}_{safe_model}.jsonl"
    out_dir = os.path.join(DATA, "results", f"article{args.article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    _MODEL_ALIASES = {"Llama-3.3-70B": "Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-Instruct-FP8": "Llama-3.3-70B"}
    done_records: dict = {}  # Filename -> serialized line
    if args.resume:
        _alias = _MODEL_ALIASES.get(safe_model)
        _paths = [out_path] + ([os.path.join(out_dir, out_name.replace(safe_model, _alias))] if _alias else [])
        for _p in _paths:
            if os.path.exists(_p):
                with open(_p) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            # Only count as done if prediction is non-null; primary path takes precedence
                            if rec.get("prediction") is not None and rec["Filename"] not in done_records:
                                done_records[rec["Filename"]] = line
                        except Exception:
                            pass
        if done_records:
            print(f"  Resuming: {len(done_records)} already done (with valid prediction)", flush=True)

    if args.limit:
        cases = cases.head(args.limit)

    # Rewrite the file cleanly: first emit all already-done records, then run remaining
    with open(out_path, "w") as fout:
        for line in done_records.values():
            fout.write(line if line.endswith("\n") else line + "\n")

        for i, (_, row) in enumerate(cases.iterrows()):
            filename = row.get("Filename", f"row_{i}")
            if filename in done_records:
                continue

            if i % 100 == 0:
                print(f"  [{i}/{len(cases)}] {filename}", flush=True)

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
