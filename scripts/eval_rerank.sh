#!/bin/bash
# Evaluate the best-saved LegalBERT re-ranker model for one article.
#
# Usage:
#   sbatch --export=ARTICLE=3,MODEL_PATH=data/vectordb/article3/rerank_model_2024-xx scripts/eval_rerank.sh
#
#SBATCH --job-name=eval_rerank
#SBATCH --output=data/data_collection/logs/eval_rerank_art%x_%j.out
#SBATCH --error=data/data_collection/logs/eval_rerank_art%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"
: "${MODEL_PATH:?Must set MODEL_PATH to the saved fold model directory}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Eval Re-Ranker — Article $ARTICLE"
echo "  Model: $MODEL_PATH"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/rerank/train.py \
    --article "$ARTICLE" \
    --model_path "$MODEL_PATH"

echo "  Done: $(date)"
echo "========================================"
