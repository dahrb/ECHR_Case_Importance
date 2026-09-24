#!/bin/bash
# Builds FAISS vector index for one article using Qwen/Qwen3-Embedding-8B.
# Run after download_qwen_embedding.sh has completed.
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/build_faiss.sh
#   sbatch --export=ARTICLE=6 scripts/build_faiss.sh
#   sbatch --export=ARTICLE=8 scripts/build_faiss.sh
#
#SBATCH --job-name=build_faiss
#SBATCH --output=data/data_collection/logs/build_faiss_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/build_faiss_art${ARTICLE}_%j.err
#SBATCH --time=12:00:00
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
echo "  Build FAISS — Article $ARTICLE"
echo "  Embedding: Qwen/Qwen3-Embedding-8B"
echo "  Chunk: 2048 / Overlap: 100"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/retrieval/initialise_vector_dbs.py \
    --article "$ARTICLE" \
    --embedding qwen \
    --chunk-size 2048 \
    --chunk-overlap 100

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
