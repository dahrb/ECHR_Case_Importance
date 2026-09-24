#!/bin/bash
# Build audited SFT JSONL locally from canonical data and existing aligned
# rationales. No inference endpoint and no SLURM dependency are required.
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
# SLURM --export treats commas as variable separators, so multi-article values
# must be passed comma-free (e.g. ARTICLES=3-6-8 or "3 6 8"). Normalise to CSV here.
ARTICLES=$(echo "$ARTICLES" | tr ' -' ',,' | tr -s ',')

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs data/finetune_v2

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Building SFT data — Articles: $ARTICLES"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/prediction/create_finetune_data.py \
    --articles "$ARTICLES" \
    --output_root data/finetune_v2

echo "========================================"
echo "  SFT data written to data/finetune_v2/"
echo "  Done: $(date)"
echo "========================================"
