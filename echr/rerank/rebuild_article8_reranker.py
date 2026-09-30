"""Train a final reranker with AP-based, group-disjoint validation."""

import argparse
import json
import math
from pathlib import Path

from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd

from echr.rerank.utils import CrossEncoderMod, extract_scalar


ROOT = Path(__file__).resolve().parents[2]


def make_samples(frame: pd.DataFrame):
    samples, groups = [], []
    for _, row in frame.iterrows():
        query = str(row["Comm_Case"]).lower()
        filename = str(row["Filename"])
        samples.append(InputExample(texts=[query, str(row["Positive"]).lower()], label=1.0))
        groups.append(filename)
        samples.append(InputExample(texts=[query, str(row["Negative"]).lower()], label=0.0))
        groups.append(filename)
    return samples, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", type=int, required=True, choices=[3, 6, 8])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-ap", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()

    vdb_dir = ROOT / "data" / "vectordb" / f"article{args.article}"
    frame = pd.read_pickle(vdb_dir / "rerank_training_data.pkl")
    if frame.empty or frame["Filename"].nunique() < 10:
        raise RuntimeError(f"Article {args.article} reranker training data is insufficient")

    samples, groups = make_samples(frame)
    filenames = list(dict.fromkeys(groups))
    train_files, valid_files = train_test_split(
        filenames, test_size=0.20, random_state=args.seed
    )
    train_set, valid_set = set(train_files), set(valid_files)
    assert train_set.isdisjoint(valid_set)
    train_samples = [sample for sample, group in zip(samples, groups) if group in train_set]
    valid_samples = [sample for sample, group in zip(samples, groups) if group in valid_set]

    print(
        f"Article {args.article} reranker: train_cases={len(train_set)} valid_cases={len(valid_set)} "
        f"train_pairs={len(train_samples)} valid_pairs={len(valid_samples)}",
        flush=True,
    )
    model = CrossEncoderMod(
        "nlpaueb/legal-bert-base-uncased", num_labels=1, classifier_dropout=0.0
    )
    loader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        valid_samples, name=f"Art{args.article}-Heldout-AP"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        vdb_dir / "rerank_model_20260924_ap_selected"
    )
    warmup_steps = math.ceil(len(loader) * args.epochs * 0.1)
    model.fit(
        train_dataloader=loader,
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=0,
        optimizer_params={"lr": args.learning_rate},
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
    )

    # Reload the AP-selected checkpoint rather than evaluating the final epoch.
    selected = CrossEncoderMod(str(output_dir), num_labels=1)
    raw_metrics = evaluator(selected)
    heldout_ap = extract_scalar(raw_metrics)
    metrics = {
        "article": args.article,
        "train_cases": len(train_set),
        "valid_cases": len(valid_set),
        "train_pairs": len(train_samples),
        "valid_pairs": len(valid_samples),
        "heldout_average_precision": heldout_ap,
        "raw_metrics": {key: float(value) for key, value in raw_metrics.items()},
        "epochs_max": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    metrics_path = output_dir / "heldout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    if heldout_ap < args.min_ap:
        raise RuntimeError(
            f"Held-out AP {heldout_ap:.4f} below required {args.min_ap:.4f}; model rejected"
        )
    print(f"ACCEPTED_RERANKER_ART{args.article}={output_dir}", flush=True)


if __name__ == "__main__":
    main()
