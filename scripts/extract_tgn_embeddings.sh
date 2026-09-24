#!/bin/bash
# Extract TGN node memory embeddings for comm_test cases (one article at a time).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/extract_tgn_embeddings.sh
#   sbatch --export=ARTICLE=6 scripts/extract_tgn_embeddings.sh
#   sbatch --export=ARTICLE=8 scripts/extract_tgn_embeddings.sh
#
#SBATCH --job-name=tgn_embed
#SBATCH --output=data/data_collection/logs/tgn_embed_art%a_%j.out
#SBATCH --error=data/data_collection/logs/tgn_embed_art%a_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Extract TGN Embeddings — Article $ARTICLE"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/graph/extract_tgn_embeddings.py --article "$ARTICLE"

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
