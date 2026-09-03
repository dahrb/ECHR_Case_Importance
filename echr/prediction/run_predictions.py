"""
Run zero-shot BASE importance predictions on comm_test cases using a local vLLM server.

Usage:
    python echr/prediction/run_predictions.py --article 3 --model gpt-oss-20b --condition base
    python echr/prediction/run_predictions.py --article 6 --model gpt-oss-20b --condition base

Output:
    data/results/article{N}/{condition}_{model}.jsonl
    Lines: {"Filename": ..., "prediction": ..., "reasoning": ..., "importance": ...}
"""

import argparse
import json
import os
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import base_zero_shot_prompt, court_prompt, COURT_MAP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_endpoint.txt")

IMPORTANCE_MAP = {"key_case": 1, "1": 2, "2": 3, "3": 4}


def get_client(endpoint: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=endpoint)


def predict_one(client: OpenAI, row, article: str, model: str, condition: str = "base",
                text: int = 1, cot: bool = False,
                max_retries: int = 3, retry_delay: float = 5.0) -> dict:
    if condition == "court":
        prompt = court_prompt(row, article, text=text)
    else:
        prompt = base_zero_shot_prompt(row, article, text=text, cot=cot)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1200,
                temperature=0.0,
                seed=42,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            raw = content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)

            if condition == "court":
                pred_raw = str(parsed.get("Court", "")).strip().lower()
                pred = COURT_MAP.get(pred_raw, None)
            else:
                pred_raw = str(parsed.get("Case Importance", "")).strip().lower()
                pred = IMPORTANCE_MAP.get(pred_raw, None)

            reasoning = parsed.get("Reasoning", "")
            return {
                "raw_prediction": pred_raw,
                "prediction": pred,
                "reasoning": reasoning,
                "raw_output": raw,
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "raw_prediction": raw if 'raw' in dir() else "",
                "prediction": None,
                "reasoning": "",
                "raw_output": raw if 'raw' in dir() else "",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay)
            else:
                print(f"  FAILED: {e}")
                return {"raw_prediction": "", "prediction": None, "reasoning": "", "raw_output": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="3, 6, or 8")
    parser.add_argument("--model", default="gpt-oss-20b", help="vLLM served model name")
    parser.add_argument("--condition", default="base", choices=["base", "court"],
                        help="Prediction condition: base (importance 1-4) or court (Judgment/Decision)")
    parser.add_argument("--text", type=int, default=1, choices=[1, 2, 3],
                        help="Text field: 1=Subject Matter, 2=Questions, 3=Both")
    parser.add_argument("--endpoint", default=None,
                        help="vLLM base URL (default: read from data/vllm_endpoint.txt)")
    parser.add_argument("--split", default="test", choices=["test", "valid"],
                        help="Which split to run on")
    parser.add_argument("--cot", action="store_true",
                        help="Chain-of-thought: add step-by-step reasoning instruction")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-predicted cases in output file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N cases (for testing)")
    args = parser.parse_args()

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(
            f"No endpoint and {ENDPOINT_FILE} not found. "
            "Start vLLM server first (scripts/start_vllm_server.sh)."
        )

    print(f"[art {args.article}] vLLM={endpoint} model={args.model} "
          f"condition={args.condition} split={args.split}")
    client = get_client(endpoint)

    split_file = "comm_test.pkl" if args.split == "test" else "comm_valid.pkl"
    cases_path = os.path.join(
        DATA, "processed", f"article{args.article}", "splits", split_file
    )
    cases = pd.read_pickle(cases_path)
    print(f"[art {args.article}] {len(cases)} cases in {args.split} split")

    safe_model = args.model.replace("/", "_").replace(":", "_")
    cot_suffix = "_cot" if args.cot else ""
    out_name = f"{args.condition}{cot_suffix}_text{args.text}_{args.split}_{safe_model}.jsonl"
    out_dir = os.path.join(DATA, "results", f"article{args.article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    done_filenames = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_filenames.add(json.loads(line)["Filename"])
                except Exception:
                    pass
        print(f"  Resuming: {len(done_filenames)} already done")

    if args.limit:
        cases = cases.head(args.limit)

    with open(out_path, "a" if args.resume else "w") as fout:
        for i, (_, row) in enumerate(cases.iterrows()):
            filename = row.get("Filename", f"row_{i}")
            if filename in done_filenames:
                continue

            if i % 100 == 0:
                print(f"  [{i}/{len(cases)}] {filename}")

            result = predict_one(client, row, args.article, args.model,
                                  condition=args.condition, text=args.text, cot=args.cot)
            record = {
                "Filename": filename,
                "importance": int(row["importance"]) if pd.notna(row.get("importance")) else None,
                **result,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[art {args.article}] predictions saved → {out_path}")


if __name__ == "__main__":
    main()
