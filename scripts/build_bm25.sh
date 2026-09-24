#!/bin/bash
# Builds BM25 retriever for one article (CPU-only, no GPU needed).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/build_bm25.sh
#   sbatch --export=ARTICLE=6 scripts/build_bm25.sh
#   sbatch --export=ARTICLE=8 scripts/build_bm25.sh
#
#SBATCH --job-name=build_bm25
#SBATCH --output=data/data_collection/logs/build_bm25_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/build_bm25_art${ARTICLE}_%j.err
#SBATCH --time=02:00:00
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
echo "  Build BM25 — Article $ARTICLE"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/retrieval/BM25.py --article "$ARTICLE"

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
