#!/bin/bash
# ============================================================
# ECHR data pipeline — Articles 6 and 8, judgment text only
#
# Runs: s6 (extract itemids) + s4 --judgment-only (scrape fact + law sections)
# Skips: s1-s5 (metadata/labels already collected), comm phase (complete)
#
# Fact section extraction is now enabled (s4_extract_text_v2_1.py line 243).
# This job will populate:
#   data/corpora/article6/{BRANCH}/fact_section/
#   data/corpora/article6/{BRANCH}/law_section/   (fills missing ~4,993 CHAMBER files)
#   data/corpora/article8/{BRANCH}/fact_section/
#   data/corpora/article8/{BRANCH}/law_section/   (fills missing ~657 CHAMBER files)
#
# Submit from repo root:
#   sbatch scripts/collect_judgment_art68.sh
# ============================================================
#SBATCH --job-name=echr_art68_judgment
#SBATCH --output=data/data_collection/logs/art68_judgment_%j.out
#SBATCH --error=data/data_collection/logs/art68_judgment_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=lowpriority

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"

cd "$REPO_DIR"

export PATH="$HOME/.local/bin:$PATH"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
PYTHON="$VENV/bin/python"

# Add repo root to PYTHONPATH so `echr` package is importable without an
# editable install (avoids writing to the venv, which may be read-only on
# compute nodes).
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p data/data_collection/logs

echo "========================================"
echo "  ECHR — Art 6 + 8 judgment text"
echo "  Started: $(date)"
echo "========================================"

"$PYTHON" data/data_collection.py \
    --steps s6 s4_judgment \
    --articles 6 8

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
