"""
Evaluate a trained LegalBERT cross-encoder re-ranker on a held-out test set.

Loads rerank_training_data.pkl for the given article, applies the same 80/20
split used during training, and scores the saved model checkpoint.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import calculate_mcc, extract_scalar

torch.cuda.empty_cache()

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--article", type=int, required=True, choices=[3, 6, 8])
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the saved CrossEncoder model directory")
args = parser.parse_args()

vdb_dir = ROOT / "data" / "vectordb" / f"article{args.article}"
train_df = pd.read_pickle(vdb_dir / "rerank_training_data.pkl")

label2int = {"neg": 0, "pos": 1}
all_samples = []
sample_groups = []
for _, row in train_df.iterrows():
    fname = str(row["Filename"])
    all_samples.append(InputExample(
        texts=[str(row["Comm_Case"]).lower(), str(row["Positive"]).lower()],
        label=label2int["pos"]
    ))
    sample_groups.append(fname)
    all_samples.append(InputExample(
        texts=[str(row["Comm_Case"]).lower(), str(row["Negative"]).lower()],
        label=label2int["neg"]
    ))
    sample_groups.append(fname)

# Case-level holdout — must match the split used in training.py
unique_files = list(dict.fromkeys(sample_groups))
_, test_files = train_test_split(unique_files, test_size=0.2, random_state=456)
test_file_set = set(test_files)
test_samples = [all_samples[i] for i, g in enumerate(sample_groups) if g in test_file_set]

model_path = args.model_path
logger.info(f"Loading model from {model_path}")
final_model = CrossEncoder(model_path)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(
    test_samples, name=f"Art{args.article}-Test"
)

score = extract_scalar(evaluator(final_model))
logger.info(f"Test AP: {score:.4f}")

y_true = [ex.label for ex in test_samples]
y_pred = [int(s > 0.5) for s in final_model.predict(evaluator.sentence_pairs)]
mcc = calculate_mcc(y_true, y_pred)
logger.info(f"Test MCC: {mcc}")

results_path = vdb_dir / "rerank_eval_results.txt"
with open(results_path, "a") as f:
    f.write(f"Model: {model_path} | Article: {args.article} | Test AP: {score:.4f} | MCC: {mcc:.4f}\n")
logger.info(f"Results written to {results_path}")
