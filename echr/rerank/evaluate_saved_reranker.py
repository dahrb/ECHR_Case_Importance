"""Evaluate an existing historical reranker on its original grouped CV fold."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", type=int, required=True, choices=[3, 6, 8])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    frame = pd.read_pickle(
        ROOT / "data" / "vectordb" / f"article{args.article}" / "rerank_training_data.pkl"
    )
    texts, labels, groups = [], [], []
    for _, row in frame.iterrows():
        query, filename = str(row["Comm_Case"]).lower(), str(row["Filename"])
        texts.extend([(query, str(row["Positive"]).lower()), (query, str(row["Negative"]).lower())])
        labels.extend([1, 0])
        groups.extend([filename, filename])

    files = list(dict.fromkeys(groups))
    cv_files, _ = train_test_split(files, test_size=0.2, random_state=456)
    cv_file_set = set(cv_files)
    cv_indices = [idx for idx, group in enumerate(groups) if group in cv_file_set]
    cv_texts = [texts[idx] for idx in cv_indices]
    cv_groups = [groups[idx] for idx in cv_indices]
    fold_splits = list(GroupKFold(n_splits=5).split(cv_texts, groups=cv_groups))
    validation_local = fold_splits[args.fold - 1][1]
    validation_indices = [cv_indices[idx] for idx in validation_local]
    validation_texts = [texts[idx] for idx in validation_indices]
    validation_labels = [labels[idx] for idx in validation_indices]

    model = CrossEncoder(args.model_dir)
    scores = model.predict(validation_texts, show_progress_bar=False)
    report = {
        "article": args.article,
        "fold": args.fold,
        "pairs": len(validation_labels),
        "average_precision": float(average_precision_score(validation_labels, scores)),
        "auc": float(roc_auc_score(validation_labels, scores)),
        "model_dir": args.model_dir,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
