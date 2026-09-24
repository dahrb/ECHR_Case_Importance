"""
Assemble SFT fine-tuning data for ECHR importance prediction.

Pipeline (mirrors old/finetune_data_creation.ipynb, generalised to Art3/6/8):

  1. Load comm_train.pkl for each article → gold labels
  2. Load base zero-shot predictions  (run_predictions.py --split train)
  3. Load iterative predictions        (run_iterative_predictions.py --split train)
  4. Find mismatch cases: both base AND iterative disagree with gold
  5. Teacher-force mismatch cases via GPT-OSS: inject gold label, ask for reasoning
  6. Assemble SFT JSONL (ChatML format): user=base prompt, assistant=gold label + best reasoning
  7. Combine all articles, shuffle, 80/20 train/val split

Output:
    data/finetune/sft_all_articles.jsonl   — full assembled dataset
    data/finetune/sft_train.jsonl          — 80 % split (used for QLoRA)
    data/finetune/sft_val.jsonl            — 20 % split (eval during training)

Usage:
    python echr/prediction/create_finetune_data.py \\
        --articles 3,6,8 \\
        --model gpt-oss-120b \\
        --endpoint http://gpu32.barkla2.liv.alces.network:8001/v1
"""

import argparse
import json
import os
import random
import time

import pandas as pd
from openai import OpenAI

from echr.prediction.prompts import base_zero_shot_prompt, IMPORTANCE_LABEL_TO_KEY

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
ENDPOINT_FILE = os.path.join(DATA, "vllm_gptoss_endpoint.txt")

IMPORTANCE_MAP = {"key_case": 1, "1": 2, "2": 3, "3": 4}


def load_base_predictions(article: str, model_safe: str) -> dict:
    """Returns {filename: {"prediction": int, "reasoning": str}}"""
    path = os.path.join(DATA, "results", f"article{article}",
                        f"base_text1_train_{model_safe}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base predictions not found: {path}")
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["Filename"]] = {
                "prediction": rec.get("prediction"),
                "reasoning": rec.get("reasoning", ""),
            }
    return out


def load_iterative_predictions(article: str, model_safe: str) -> dict:
    """Returns {filename: {"prediction": int, "reasoning": str}}"""
    path = os.path.join(DATA, "results", f"article{article}",
                        f"iterative_text1_train_{model_safe}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Iterative predictions not found: {path}")
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            best_level = rec.get("predicted_level", "")
            # Extract reasoning for the winning level
            level_results = rec.get("level_results", {})
            reasoning = level_results.get(best_level, {}).get("reasoning", "")
            out[rec["Filename"]] = {
                "prediction": rec.get("prediction"),
                "reasoning": reasoning,
            }
    return out


def teacher_force_one(client: OpenAI, row, article: str, model: str,
                       gold_int: int, max_retries: int = 3, retry_delay: float = 5.0) -> str:
    """Ask the LLM to justify the gold label. Returns reasoning string."""
    gold_key = IMPORTANCE_LABEL_TO_KEY[gold_int]
    base = base_zero_shot_prompt(row, article, text=1)
    prompt = (
        base
        + f"\n\nThe importance level of this case is: {gold_key}. "
        f"Please only predict this importance level and provide reasons why "
        f"the case is importance level {gold_key}."
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
                seed=42,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return str(parsed.get("Reasoning", raw))
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt+1}/{max_retries}: {e}", flush=True)
                time.sleep(retry_delay)
            else:
                print(f"  Teacher-force FAILED: {e}", flush=True)
                return ""
    return ""


def build_article_sft(article: str, client: OpenAI, model: str) -> list[dict]:
    """Build SFT records for one article. Returns list of ChatML message dicts."""
    model_safe = model.replace("/", "_").replace(":", "_")

    cases = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "splits", "comm_train.pkl")
    )
    print(f"\n[art{article}] {len(cases)} train cases", flush=True)

    base_preds = load_base_predictions(article, model_safe)
    iter_preds = load_iterative_predictions(article, model_safe)

    records = []
    mismatch_count = 0
    source_counts = {"iter": 0, "base": 0, "mismatch": 0, "failed": 0}

    for _, row in cases.iterrows():
        fname = row["Filename"]
        gold = int(row["importance"])

        bp = base_preds.get(fname, {})
        ip = iter_preds.get(fname, {})
        base_pred = bp.get("prediction")
        iter_pred = ip.get("prediction")

        # Priority: iterative if correct → base if correct → teacher-force mismatch
        if iter_pred == gold:
            reasoning = ip.get("reasoning", "")
            source = "iter"
        elif base_pred == gold:
            reasoning = bp.get("reasoning", "")
            source = "base"
        else:
            # Both wrong: teacher-force to get gold-label reasoning
            mismatch_count += 1
            print(f"  Mismatch {fname}: gold={gold} iter={iter_pred} base={base_pred} → teacher-forcing", flush=True)
            reasoning = teacher_force_one(client, row, article, model, gold)
            source = "mismatch" if reasoning else "failed"

        source_counts[source] += 1
        gold_key = IMPORTANCE_LABEL_TO_KEY[gold]
        user_prompt = base_zero_shot_prompt(row, article, text=1)
        assistant_content = json.dumps({"Case Importance": gold_key, "Reasoning": reasoning})

        records.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_content},
            ],
            "_meta": {"article": article, "filename": fname, "gold": gold, "source": source},
        })

    print(f"[art{article}] Sources: {source_counts} (mismatches: {mismatch_count})", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", default="3,6,8",
                        help="Comma-separated articles to include (default: 3,6,8)")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None,
                        help="Output subdir under data/finetune/ (default: auto — "
                             "'art3' for single article, 'combined' for multiple)")
    args = parser.parse_args()

    if args.endpoint:
        endpoint = args.endpoint
    elif os.path.exists(ENDPOINT_FILE):
        endpoint = open(ENDPOINT_FILE).read().strip()
    else:
        raise RuntimeError(f"No endpoint and {ENDPOINT_FILE} not found.")

    client = OpenAI(api_key="EMPTY", base_url=endpoint)
    articles = [a.strip() for a in args.articles.split(",")]

    all_records = []
    for article in articles:
        all_records.extend(build_article_sft(article, client, args.model))

    print(f"\nTotal SFT records: {len(all_records)}", flush=True)

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(all_records)
    n_val = max(1, int(len(all_records) * args.val_frac))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    # Output subdir: single article → 'artN', multiple → 'combined' (or user --tag)
    if args.tag:
        tag = args.tag
    elif len(articles) == 1:
        tag = f"art{articles[0]}"
    else:
        tag = "combined"
    out_dir = os.path.join(DATA, "finetune", tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output tag: '{tag}' → {out_dir}", flush=True)

    def write_jsonl(path, records):
        # Strip _meta before writing (not needed by trainer)
        with open(path, "w") as f:
            for rec in records:
                out = {"messages": rec["messages"]}
                f.write(json.dumps(out) + "\n")
        print(f"  Written {len(records)} records → {path}", flush=True)

    # Full dataset (with meta, for inspection)
    full_path = os.path.join(out_dir, "sft_all_articles.jsonl")
    with open(full_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nFull dataset ({len(all_records)} records) → {full_path}", flush=True)

    write_jsonl(os.path.join(out_dir, "sft_train.jsonl"), train_records)
    write_jsonl(os.path.join(out_dir, "sft_val.jsonl"), val_records)

    print(f"\nDone. Train={len(train_records)} Val={len(val_records)}", flush=True)


if __name__ == "__main__":
    main()
