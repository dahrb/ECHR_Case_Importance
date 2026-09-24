#!/bin/bash
# Run few-shot importance predictions (one fixed example per importance class from comm_valid).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/predict_few_shot.sh
#   sbatch --export=ARTICLE=6,MODEL=gpt-oss-120b scripts/predict_few_shot.sh
#
# Optional overrides:
#   MODEL: vLLM model name (default gpt-oss-120b)
#   TEXT: 1=Subject Matter (default), 2=Questions, 3=Both
#   SPLIT: test (default) or valid
#
#SBATCH --job-name=predict_fewshot
#SBATCH --output=data/data_collection/logs/predict_fewshot_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/predict_fewshot_art${ARTICLE}_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

MODEL="${MODEL:-gpt-oss-120b}"
TEXT="${TEXT:-1}"
SPLIT="${SPLIT:-test}"
ENDPOINT="${ENDPOINT:-}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs data/results/article"$ARTICLE"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "  Few-Shot Predictions — Article $ARTICLE"
echo "  model=$MODEL  text=$TEXT  split=$SPLIT"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/prediction/run_few_shot_predictions.py \
    --article "$ARTICLE" \
    --model "$MODEL" \
    --text "$TEXT" \
    --split "$SPLIT" \
    --resume \
    ${ENDPOINT:+--endpoint "$ENDPOINT"}

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
