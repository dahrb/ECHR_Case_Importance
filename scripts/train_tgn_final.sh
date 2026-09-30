#!/bin/bash
# Refit the paper-faithful feature-free retrieval TGN on all citation events.
# The epoch count is selected by the preceding chronological 70/15/15 run.
# Usage: sbatch --export=ALL,ARTICLE=3 scripts/train_tgn_final.sh
#SBATCH --job-name=tgn_final
#SBATCH --output=data/data_collection/logs/tgn_final_art%a_%j.out
#SBATCH --error=data/data_collection/logs/tgn_final_art%a_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s-low

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8)}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
NETWORK_DIR="$REPO_DIR/data/network/article$ARTICLE"
EVAL_CHECKPOINT="$NETWORK_DIR/best_model_False.pth"
EVAL_ARCHIVE="$NETWORK_DIR/best_model_False_eval.pth"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Preserve the validation-selected checkpoint and metrics before the full-data
# refit writes the retrieval checkpoint at the historical path.
cp -p "$EVAL_CHECKPOINT" "$EVAL_ARCHIVE"
BEST_EPOCH=$("$VENV/bin/python" -c \
  'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"])' \
  "$EVAL_ARCHIVE")

echo "========================================"
echo "  Final full-data TGN — Article $ARTICLE"
echo "  selected_epochs=$BEST_EPOCH node_features=0"
echo "  eval_checkpoint=$EVAL_ARCHIVE"
echo "  retrieval_checkpoint=$EVAL_CHECKPOINT"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/graph/network_TGN.py \
  --article "$ARTICLE" \
  --epochs "$BEST_EPOCH" \
  --n_runs 1 \
  --seed 42 \
  --no_node_features_inc \
  --no_val_test

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
