#!/bin/bash
# Runs FAISS retrieval experiments for one article (Qwen3-Embedding-8B).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/run_retrieval_faiss.sh
#   sbatch --export=ARTICLE=6 scripts/run_retrieval_faiss.sh
#   sbatch --export=ARTICLE=8 scripts/run_retrieval_faiss.sh
#
#SBATCH --job-name=run_faiss
#SBATCH --output=data/data_collection/logs/run_faiss_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/run_faiss_art${ARTICLE}_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
MODEL_CACHE="$REPO_DIR/models/hub"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Run FAISS Retrieval — Article $ARTICLE"
echo "  Embedding: Qwen/Qwen3-Embedding-8B"
echo "  Similarity: cosine"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/retrieval/run_experiments.py \
    --article "$ARTICLE" \
    --embedding-name "Qwen/Qwen3-Embedding-8B" \
    --short-name "qwen3-8b_raw" \
    --chunk-size 2048 \
    --similarity cosine

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
