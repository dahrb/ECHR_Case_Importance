#!/bin/bash
# Train LegalBERT cross-encoder re-ranker for one article (5-fold CV).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/train_rerank.sh
#
# Optional overrides (comma-separated in --export):
#   LR=1e-5, BATCH=16, DROPOUT=0.0, EPOCHS=30
#
#SBATCH --job-name=train_rerank
#SBATCH --output=data/data_collection/logs/train_rerank_art%x_%j.out
#SBATCH --error=data/data_collection/logs/train_rerank_art%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

LR="${LR:-1e-5}"
BATCH="${BATCH:-16}"
DROPOUT="${DROPOUT:-0.0}"
EPOCHS="${EPOCHS:-30}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Train Re-Ranker — Article $ARTICLE"
echo "  lr=$LR batch=$BATCH dropout=$DROPOUT epochs=$EPOCHS"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/rerank/training.py \
    --article "$ARTICLE" \
    --learning_rate "$LR" \
    --batch_size "$BATCH" \
    --dropout "$DROPOUT" \
    --epochs "$EPOCHS"

echo "  Done: $(date)"
echo "========================================"
