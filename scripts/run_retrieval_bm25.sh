#!/bin/bash
# Runs BM25 retrieval experiments for one article (CPU-only).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/run_retrieval_bm25.sh
#   sbatch --export=ARTICLE=6 scripts/run_retrieval_bm25.sh
#   sbatch --export=ARTICLE=8 scripts/run_retrieval_bm25.sh
#
#SBATCH --job-name=run_bm25
#SBATCH --output=data/data_collection/logs/run_bm25_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/run_bm25_art${ARTICLE}_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Run BM25 Retrieval — Article $ARTICLE"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/retrieval/run_experiments.py \
    --article "$ARTICLE" \
    --embedding-name "BM25" \
    --short-name "bm25"

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
