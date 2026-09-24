"""Build leakage-free, per-article SFT datasets from the canonical splits.

The original notebook's alignment rule is retained: use reasoning only when it
supports the gold label. ``comm_train.pkl`` is used for training,
``comm_valid.pkl`` for validation, and ``comm_test.pkl`` remains untouched. If
the stored validation file overlaps train or test (currently true for Article
3), a deterministic, class-aware holdout is made from the current training
pool instead. No combined dataset and no new reasoning traces are produced.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from echr.prediction.prompts import IMPORTANCE_LABEL_TO_KEY, base_zero_shot_prompt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(article: str, split: str, kind: str) -> dict[str, dict[str, Any]]:
    result_dir = DATA / "results" / f"article{article}"
    if kind == "base":
        path = result_dir / f"base_text1_{split}_gpt-oss-120b.jsonl"
    elif kind == "iterative":
        path = result_dir / f"iterative_text1_{split}_gpt-oss-120b.jsonl"
    else:
        raise ValueError(kind)

    output: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        reasoning = record.get("reasoning", "") or ""
        if kind == "iterative":
            level = record.get("predicted_level", "")
            reasoning = record.get("level_results", {}).get(level, {}).get("reasoning", "") or ""
        output[str(record["Filename"])] = {
            "prediction": record.get("prediction"),
            "reasoning": str(reasoning).strip(),
        }
    return output


def load_aligned_cache(article: str) -> dict[str, dict[str, Any]]:
    """Load rationales produced by the old correct/base/teacher-force builder."""
    path = DATA / "finetune" / f"art{article}" / "sft_all_articles.jsonl"
    cache: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        meta = record.get("_meta", {})
        filename = str(meta.get("filename", ""))
        if not filename:
            continue
        try:
            answer = json.loads(record["messages"][-1]["content"])
            gold = int(meta["gold"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        cache[filename] = {
            "gold": gold,
            "reasoning": str(answer.get("Reasoning", "") or "").strip(),
            "source": str(meta.get("source", "cached")),
        }
    return cache


def choose_reasoning(filename: str, gold: int, split: str,
                     cache: dict[str, dict[str, Any]],
                     iterative: dict[str, dict[str, Any]],
                     base: dict[str, dict[str, Any]]) -> tuple[str, str]:
    cached = cache.get(filename)
    if split == "train" and cached and cached["gold"] == gold and cached["reasoning"]:
        return cached["reasoning"], f"aligned_cache_{cached['source']}"

    prediction = iterative.get(filename, {})
    if prediction.get("prediction") == gold and prediction.get("reasoning"):
        return str(prediction["reasoning"]), "iterative_correct"

    prediction = base.get(filename, {})
    if prediction.get("prediction") == gold and prediction.get("reasoning"):
        return str(prediction["reasoning"]), "base_correct"

    # Never attach a rationale produced for a different label.
    return "", "gold_label_only"


def build_split(article: str, cases: pd.DataFrame, prediction_split: str,
                output_split: str, cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base = load_predictions(article, prediction_split, "base")
    iterative = load_predictions(article, prediction_split, "iterative")
    records = []
    for _, row in cases.iterrows():
        filename = str(row["Filename"])
        gold = int(row["importance"])
        gold_label = IMPORTANCE_LABEL_TO_KEY[gold]
        reasoning, source = choose_reasoning(
            filename, gold, prediction_split, cache, iterative, base
        )
        answer = {"Case Importance": gold_label, "Reasoning": reasoning}
        records.append({
            "messages": [
                {"role": "user", "content": base_zero_shot_prompt(row, article, text=1)},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
            ],
            "metadata": {
                "article": int(article), "filename": filename, "gold": gold,
                "gold_label": gold_label, "reasoning_source": source,
                "split": output_split, "source_split": prediction_split,
            },
        })
    return records


def audit(article: str, train: list[dict[str, Any]], validation: list[dict[str, Any]],
          test_ids: set[str], split_note: dict[str, Any]) -> dict[str, Any]:
    train_ids = [record["metadata"]["filename"] for record in train]
    valid_ids = [record["metadata"]["filename"] for record in validation]
    if len(train_ids) != len(set(train_ids)) or len(valid_ids) != len(set(valid_ids)):
        raise ValueError(f"Article {article}: duplicate filenames within a split")
    overlap = sorted(set(train_ids) & set(valid_ids))
    if overlap:
        raise ValueError(f"Article {article}: train/validation leakage: {overlap[:10]}")
    train_test_overlap = sorted(set(train_ids) & test_ids)
    valid_test_overlap = sorted(set(valid_ids) & test_ids)
    if train_test_overlap or valid_test_overlap:
        raise ValueError(
            f"Article {article}: test leakage: train={train_test_overlap[:10]}, "
            f"validation={valid_test_overlap[:10]}"
        )

    report: dict[str, Any] = {
        "article": int(article), "train_validation_overlap": 0,
        "train_test_overlap": 0, "validation_test_overlap": 0,
        "validation_split": split_note,
    }
    for name, records in (("train", train), ("validation", validation)):
        labels = Counter(record["metadata"]["gold_label"] for record in records)
        sources = Counter(record["metadata"]["reasoning_source"] for record in records)
        empty = sum(not json.loads(record["messages"][-1]["content"])["Reasoning"]
                    for record in records)
        report[name] = {
            "examples": len(records), "labels": dict(sorted(labels.items())),
            "reasoning_sources": dict(sorted(sources.items())), "empty_reasoning": empty,
        }
    return report


def prepare_splits(article: str, seed: int = 42,
                   holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame,
                                                               set[str], dict[str, Any]]:
    split_dir = DATA / "processed" / f"article{article}" / "splits"
    train = pd.read_pickle(split_dir / "comm_train.pkl").copy()
    validation = pd.read_pickle(split_dir / "comm_valid.pkl").copy()
    test = pd.read_pickle(split_dir / "comm_test.pkl")
    ids = lambda frame: set(frame["Filename"].astype(str))
    train_valid = sorted(ids(train) & ids(validation))
    valid_test = sorted(ids(validation) & ids(test))

    if not train_valid and not valid_test:
        note = {"method": "canonical_comm_valid", "seed": None,
                "holdout_fraction": None, "rejected_overlap_ids": []}
        return train, validation, ids(test), note

    # The stale Article 3 validation file overlaps both current train and test.
    # Make a reproducible holdout, independently within each class, so even the
    # rare labels are represented whenever at least two examples exist.
    rng = random.Random(seed)
    validation_indices: list[Any] = []
    for _, group in train.groupby("importance", sort=True):
        candidates = list(group.index)
        rng.shuffle(candidates)
        count = 1 if len(candidates) > 1 else 0
        count = max(count, round(len(candidates) * holdout_fraction))
        count = min(count, len(candidates) - 1) if len(candidates) > 1 else 0
        validation_indices.extend(candidates[:count])
    validation_indices = sorted(validation_indices)
    validation = train.loc[validation_indices].copy()
    train = train.drop(index=validation_indices).copy()
    note = {
        "method": "class_aware_holdout_from_current_comm_train",
        "seed": seed, "holdout_fraction": holdout_fraction,
        "reason": "stored comm_valid overlaps current train and/or test",
        "stored_train_validation_overlap": train_valid,
        "stored_validation_test_overlap": valid_test,
        "held_out_ids": sorted(ids(validation)),
    }
    return train, validation, ids(test), note


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", default="3,6,8")
    parser.add_argument("--output_root", default="data/finetune_v2")
    args = parser.parse_args()
    articles = [x.strip() for x in args.articles.replace("-", ",").split(",") if x.strip()]
    if any(article not in {"3", "6", "8"} for article in articles):
        raise ValueError("Only Articles 3, 6 and 8 are supported")

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for article in articles:
        cache = load_aligned_cache(article)
        train_cases, validation_cases, test_ids, split_note = prepare_splits(article)
        train = build_split(article, train_cases, "train", "train", cache)
        validation_prediction_split = (
            "train" if split_note["method"].startswith("class_aware") else "valid"
        )
        validation = build_split(
            article, validation_cases, validation_prediction_split, "validation", cache
        )
        report = audit(article, train, validation, test_ids, split_note)
        reports.append(report)
        destination = output_root / f"art{article}"
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / "sft_train.jsonl", train)
        write_jsonl(destination / "sft_val.jsonl", validation)
        (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    (output_root / "audit_all.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
