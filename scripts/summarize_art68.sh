#!/bin/bash
# Generate 200- and 500-word GPT-OSS summaries for Art 6 and Art 8 outcome cases.
# Requires the vLLM server to be running (start with scripts/start_vllm_server.sh).
#
# Usage:
#   sbatch --export=ARTICLE=6 scripts/summarize_art68.sh
#   sbatch --export=ARTICLE=8 scripts/summarize_art68.sh
#
# Optional:
#   --export=ARTICLE=6,MODEL=gpt-oss-20b
#
#SBATCH --job-name=summarize
#SBATCH --output=data/data_collection/logs/summarize_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/summarize_art${ARTICLE}_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (6 or 8) via --export=ARTICLE=N}"

MODEL="${MODEL:-gpt-oss-20b}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "  Summarize Art $ARTICLE Outcome Cases"
echo "  model=$MODEL"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/prediction/summarize_cases.py \
    --article "$ARTICLE" \
    --model "$MODEL" \
    --resume

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
