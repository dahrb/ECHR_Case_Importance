#!/bin/bash
# Trains the TGN citation-link-prediction model for one article.
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/train_tgn.sh
#   sbatch --export=ARTICLE=6 scripts/train_tgn.sh
#   sbatch --export=ARTICLE=8 scripts/train_tgn.sh
#
# Optional overrides via --export (comma-separated), e.g.:
#   sbatch --export=ARTICLE=3,EPOCHS=20,NODE_FEATURES=1 scripts/train_tgn.sh
#
#SBATCH --job-name=train_tgn
#SBATCH --output=data/data_collection/logs/train_tgn_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/train_tgn_art${ARTICLE}_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

EPOCHS="${EPOCHS:-20}"
N_RUNS="${N_RUNS:-1}"
SEED="${SEED:-42}"
NODE_FEATURES="${NODE_FEATURES:-1}"   # 1 = include node features, 0 = memory only

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$NODE_FEATURES" = "1" ]; then
    NF_FLAG="--node_features_inc"
else
    NF_FLAG="--no_node_features_inc"
fi

echo "========================================"
echo "  Train TGN — Article $ARTICLE"
echo "  epochs=$EPOCHS n_runs=$N_RUNS seed=$SEED node_features=$NODE_FEATURES"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/graph/network_TGN.py \
    --article "$ARTICLE" \
    --epochs "$EPOCHS" \
    --n_runs "$N_RUNS" \
    --seed "$SEED" \
    $NF_FLAG \
    --val_test

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
