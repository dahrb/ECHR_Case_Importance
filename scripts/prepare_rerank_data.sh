#!/bin/bash
# Build re-ranker training data for one article.
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/prepare_rerank_data.sh
#
#SBATCH --job-name=rerank_prep
#SBATCH --output=data/data_collection/logs/rerank_prep_art%x_%j.out
#SBATCH --error=data/data_collection/logs/rerank_prep_art%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Prepare Re-Ranker Training Data — Article $ARTICLE"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/rerank/prepare_data.py --article "$ARTICLE"

echo "  Done: $(date)"
echo "========================================"
