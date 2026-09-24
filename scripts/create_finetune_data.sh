#!/bin/bash
# Build SFT fine-tuning JSONL from GPT-OSS train-split predictions.
# Includes mismatch teacher-forcing (small number of LLM calls).
# Depends on: run_gptoss_train_inference.sh jobs completing first.
#
# Usage:
#   sbatch scripts/create_finetune_data.sh             # all articles
#   sbatch --export=ARTICLES=3 scripts/create_finetune_data.sh   # Art3 only
#
#SBATCH --job-name=create_ft_data
#SBATCH --output=data/data_collection/logs/create_ft_data_%j.out
#SBATCH --error=data/data_collection/logs/create_ft_data_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

ARTICLES="${ARTICLES:-3,6,8}"
MODEL="${MODEL:-gpt-oss-120b}"
ENDPOINT="${ENDPOINT:-}"
TAG="${TAG:-}"

# SLURM --export treats commas as variable separators, so multi-article values
# must be passed comma-free (e.g. ARTICLES=3-6-8 or "3 6 8"). Normalise to CSV here.
ARTICLES=$(echo "$ARTICLES" | tr ' -' ',,' | tr -s ',')

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs data/finetune

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Building SFT data — Articles: $ARTICLES"
echo "  Model: $MODEL"
echo "  Started: $(date)"
echo "========================================"

# Use endpoint from file if not explicitly set
if [ -z "$ENDPOINT" ]; then
    ENDPOINT=$(cat data/vllm_gptoss_endpoint.txt)
fi

"$VENV/bin/python" echr/prediction/create_finetune_data.py \
    --articles "$ARTICLES" \
    --model "$MODEL" \
    --endpoint "$ENDPOINT" \
    ${TAG:+--tag "$TAG"}

echo "========================================"
echo "  SFT data written to data/finetune/"
echo "  Done: $(date)"
echo "========================================"
