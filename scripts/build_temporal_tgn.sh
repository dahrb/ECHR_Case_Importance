#!/bin/bash
# Build temporally truncated, VDB-seeded TGN KG retrieval results.
# Usage: sbatch --export=ALL,ARTICLE=3 scripts/build_temporal_tgn.sh
#SBATCH --job-name=tgn_kg_build
#SBATCH --output=data/data_collection/logs/tgn_kg_build_art%a_%j.out
#SBATCH --error=data/data_collection/logs/tgn_kg_build_art%a_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s-low

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8)}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$VENV/bin/python" echr/retrieval/build_tgn_results.py \
    --article "$ARTICLE" \
    --seed-k 3 5 10 \
    --keep 100
