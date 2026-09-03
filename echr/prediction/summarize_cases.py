"""
Generate 200- and 500-word summaries for outcome cases using a local vLLM server.

Usage:
    python echr/prediction/summarize_cases.py --article 6 --model gpt-oss-20b
    python echr/prediction/summarize_cases.py --article 8 --model gpt-oss-20b

Output:
    data/processed/article{N}/outcome_summaries.pkl
    (DataFrame with columns: Filename, 200 Word Summary, 500 Word Summary)
"""

import argparse
import json
import re
import os
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import summarize_outcome_prompt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_endpoint.txt")


def get_client(endpoint: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(api_key=api_key, base_url=endpoint)


def _make_truncated_row(row, max_chars_each: int):
    """Return a copy of row with Facts and The Law truncated to max_chars_each."""
    import copy
    r = dict(row)
    for field in ("Facts", "The Law"):
        val = r.get(field, "")
        if val and len(str(val)) > max_chars_each:
            r[field] = str(val)[:max_chars_each] + " [truncated]"
    return r


def summarize_one(client: OpenAI, row, article: str, model: str,
                  max_retries: int = 3, retry_delay: float = 5.0) -> dict:
    # truncation budgets per retry: full → ~40K chars each → ~20K chars each
    truncation_budgets = [None, 40_000, 20_000]
    active_row = row
    for attempt in range(max_retries):
        if truncation_budgets[attempt] is not None:
            active_row = _make_truncated_row(row, truncation_budgets[attempt])
            print(f"  Retrying with truncated text ({truncation_budgets[attempt]} chars/field)", flush=True)
        prompt = summarize_outcome_prompt(active_row, article)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3500,
                temperature=0.0,
                seed=42,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            raw = content.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            # GPT-OSS may emit thinking tokens before JSON — find first { block
            if not raw.startswith("{"):
                match = re.search(r"\{", raw)
                if match:
                    raw = raw[match.start():]
            parsed = json.loads(raw)
            return {
                "200 Word Summary": parsed.get("200 Word Summary", ""),
                "500 Word Summary": parsed.get("500 Word Summary", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return {"200 Word Summary": raw, "500 Word Summary": raw}
        except Exception as e:
            err_str = str(e)
            # context-overflow: don't sleep, just immediately retry with smaller text
            if "max_tokens must be at least 1" in err_str or "context_length_exceeded" in err_str:
                print(f"  Context overflow (attempt {attempt+1}): {e}", flush=True)
                # next attempt will pick a smaller budget from truncation_budgets
                continue
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}", flush=True)
                return {"200 Word Summary": "", "500 Word Summary": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="3, 6, or 8")
    parser.add_argument("--model", default="gpt-oss-20b", help="vLLM served model name")
    parser.add_argument("--endpoint", default=None,
                        help="vLLM base URL (default: read from data/vllm_endpoint.txt)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N cases (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases already in output file")
    args = parser.parse_args()

    # resolve endpoint
    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(
            f"No endpoint provided and {ENDPOINT_FILE} not found. "
            "Start the vLLM server first (scripts/start_vllm_server.sh)."
        )

    print(f"[art {args.article}] connecting to vLLM at {endpoint} | model={args.model}", flush=True)
    client = get_client(endpoint)

    cases_path = os.path.join(DATA, "processed", f"article{args.article}", "outcome_cases.pkl")
    outcome_cases = pd.read_pickle(cases_path)
    print(f"[art {args.article}] {len(outcome_cases)} outcome cases", flush=True)

    out_path = os.path.join(DATA, "processed", f"article{args.article}", "outcome_summaries.pkl")

    existing = {}
    if args.resume and os.path.exists(out_path):
        prev = pd.read_pickle(out_path)
        existing = {row["Filename"]: row.to_dict() for _, row in prev.iterrows()}
        print(f"[art {args.article}] resuming — {len(existing)} already done", flush=True)

    if args.limit:
        outcome_cases = outcome_cases.head(args.limit)

    records = []
    for i, (_, row) in enumerate(outcome_cases.iterrows()):
        filename = row.get("Filename") or row.get("File", f"row_{i}")
        if filename in existing:
            records.append(existing[filename])
            continue

        if i % 100 == 0:
            print(f"  [{i}/{len(outcome_cases)}] {filename}", flush=True)

        result = summarize_one(client, row, args.article, args.model)
        records.append({"Filename": filename, **result})

        # checkpoint every 500 cases
        if len(records) % 500 == 0:
            pd.to_pickle(pd.DataFrame(records), out_path)

    df_out = pd.DataFrame(records)
    pd.to_pickle(df_out, out_path)
    print(f"[art {args.article}] saved {len(df_out)} summaries → {out_path}", flush=True)


if __name__ == "__main__":
    main()
