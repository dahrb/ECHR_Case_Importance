import argparse
import math
import logging
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import torch
from sklearn.model_selection import GroupKFold, train_test_split
import pandas as pd

from utils import CrossEncoderMod, extract_scalar

torch.cuda.empty_cache()

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--article", type=int, required=True, choices=[3, 6, 8])
parser.add_argument("--dropout", "-d", type=float, default=0.0)
parser.add_argument("--learning_rate", "-l", type=float, default=1e-5)
parser.add_argument("--batch_size", "-b", type=int, default=16)
parser.add_argument("--epochs", type=int, default=30)
(args, _) = parser.parse_known_args()

model_name = "nlpaueb/legal-bert-base-uncased"
num_epochs = args.epochs
article = args.article

vdb_dir = ROOT / "data" / "vectordb" / f"article{article}"
model_save_path = str(vdb_dir / f"rerank_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

train_df = pd.read_pickle(vdb_dir / "rerank_training_data.pkl")

label2int = {"neg": 0, "pos": 1}
all_samples = []
sample_groups = []  # parallel list: Filename for each InputExample
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

# Case-level 80/20 holdout: split by unique Filename so no case straddles train/test
unique_files = list(dict.fromkeys(sample_groups))
cv_files, _ = train_test_split(unique_files, test_size=0.2, random_state=456)
cv_file_set = set(cv_files)
train_indices = [i for i, g in enumerate(sample_groups) if g in cv_file_set]
train_samples = [all_samples[i] for i in train_indices]
train_groups = [sample_groups[i] for i in train_indices]

gkf = GroupKFold(n_splits=5)

param_grid = {
    "learning_rate": [args.learning_rate],
    "batch_size": [args.batch_size],
    "dropout": [args.dropout],
}

best_ap_score = 0
best_params = None

for lr in param_grid["learning_rate"]:
    for bs in param_grid["batch_size"]:
        for dropout in param_grid["dropout"]:
            logger.info(f"Training with lr={lr}, batch_size={bs}, dropout={dropout}")
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(gkf.split(train_samples, groups=train_groups)):
                logger.info(f"Fold {fold + 1}")

                model = CrossEncoderMod(model_name, num_labels=1, classifier_dropout=dropout)

                train_subset = Subset(train_samples, train_idx)
                val_subset = Subset(train_samples, val_idx)
                train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=bs)

                warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
                logger.info(f"Warmup steps: {warmup_steps}")

                evaluator = CEBinaryClassificationEvaluator.from_input_examples(
                    val_subset, name=f"Art{article}-Relevance"
                )

                fold_path = f"{model_save_path}_fold_{fold + 1}"
                model.fit(
                    train_dataloader=train_dataloader,
                    evaluator=evaluator,
                    epochs=num_epochs,
                    evaluation_steps=0,
                    optimizer_params={"lr": lr},
                    warmup_steps=warmup_steps,
                    output_path=fold_path,
                )

                score = extract_scalar(evaluator(model))
                logger.info(f"Fold {fold + 1} AP: {score}")
                fold_results.append(score)

            avg_ap = sum(fold_results) / len(fold_results)
            logger.info(f"Avg AP for lr={lr}, bs={bs}, dropout={dropout}: {avg_ap}")

            if avg_ap > best_ap_score:
                best_ap_score = avg_ap
                best_params = {"learning_rate": lr, "batch_size": bs, "dropout": dropout}

logger.info(f"Best params: {best_params}, avg AP: {best_ap_score}")

results_path = vdb_dir / "rerank_training_results.txt"
with open(results_path, "a") as f:
    f.write(f"{datetime.now()} | Article {article} | Params: {best_params} | Avg AP: {best_ap_score}\n")

logger.info(f"Results written to {results_path}")
