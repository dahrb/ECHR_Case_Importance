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
import re
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import base_zero_shot_prompt, court_prompt, COURT_MAP, COURT_SOURCE_MAP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_endpoint.txt")

IMPORTANCE_MAP = {"key_case": 1, "1": 2, "2": 3, "3": 4}


def get_client(endpoint: str) -> OpenAI:
    return OpenAI(api_key="EMPTY", base_url=endpoint)


def predict_one(client: OpenAI, row, article: str, model: str, condition: str = "base",
                text: int = 1, cot: bool = False, max_tokens: int = 1200,
                max_model_len: int = 32000,
                reasoning_effort: str = None,
                no_thinking: bool = False,
                max_retries: int = 3, retry_delay: float = 5.0) -> dict:
    # Budget: model context - output tokens - prompt template overhead (~500 tok)
    # Use 3 chars/token (conservative for legal text) + 10% safety margin
    max_chars = max(5000, int((max_model_len - max_tokens - 500) * 0.90) * 3 - 1500)
    if condition == "court":
        prompt = court_prompt(row, article, text=text, max_chars=max_chars)
    else:
        prompt = base_zero_shot_prompt(row, article, text=text, cot=cot, max_chars=max_chars)

    extra = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    # For LoRA adapter models: disable thinking entirely — RE=medium is not enforced by
    # vLLM through --enable-lora, so the model exhausts its budget on thinking tokens.
    if no_thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                seed=42,
                extra_body=extra,
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

            if condition == "court":
                pred_raw = str(parsed.get("Court", "")).strip().lower()
                # Try exact match first, then prefix match for "grand chamber" variants
                pred = COURT_MAP.get(pred_raw, None)
                if pred is None:
                    for key, val in COURT_MAP.items():
                        if pred_raw.startswith(key) or key in pred_raw:
                            pred = val
                            break
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
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"],
                        help="Which split to run on")
    parser.add_argument("--cot", action="store_true",
                        help="Chain-of-thought: add step-by-step reasoning instruction")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-predicted cases in output file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N cases (for testing)")
    parser.add_argument("--max_tokens", type=int, default=1200,
                        help="Generation budget (raise for reasoning models like gpt-oss FT)")
    parser.add_argument("--max_model_len", type=int, default=32000,
                        help="Server max_model_len; used to compute prompt budget")
    parser.add_argument("--reasoning_effort", default=None,
                        choices=["low", "medium", "high"],
                        help="gpt-oss reasoning effort; use 'medium' for base model")
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable thinking entirely (for LoRA FT models where RE=medium is not enforced)")
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

    split_file = {"test": "comm_test.pkl", "valid": "comm_valid.pkl", "train": "comm_train.pkl"}[args.split]
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

    _MODEL_ALIASES = {"Llama-3.3-70B": "Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-Instruct-FP8": "Llama-3.3-70B"}
    done_filenames = set()
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
                                  condition=args.condition, text=args.text, cot=args.cot,
                                  max_tokens=args.max_tokens,
                                  max_model_len=args.max_model_len,
                                  reasoning_effort=args.reasoning_effort,
                                  no_thinking=args.no_thinking)
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
