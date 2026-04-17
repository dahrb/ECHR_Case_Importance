#!/bin/bash
# ============================================================
# ECHR data pipeline — Article 8
# Runs s1 → s2 → s3 → s4 → s5 for articles/art_8
#
# Submit with:
#   sbatch jobs/pipeline_art8.sh
# ============================================================
#SBATCH --job-name=echr_art8
#SBATCH --output=jobs/logs/art8_%j.out
#SBATCH --error=jobs/logs/art8_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=lowpriority

set -euo pipefail

ARTICLE_DIR="articles/art_8"
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"

cd "$REPO_DIR"

# Ensure uv is on PATH (it lives in ~/.local/bin on this cluster)
export PATH="$HOME/.local/bin:$PATH"

mkdir -p Data/logs

echo "========================================"
echo "  ECHR Pipeline — Article 8"
echo "  Started: $(date)"
echo "========================================"

# ----------------------------------------------------------
# Step 1: Fetch raw metadata from HUDOC for Art. 8 queries
# ----------------------------------------------------------
echo "[s1] Fetching metadata..."
uv run python Data/s1_extract_meta_v1_1.py --article_dir "$ARTICLE_DIR"
echo "[s1] Done: $(date)"

# ----------------------------------------------------------
# Step 2: Overlap / deduplication
# ----------------------------------------------------------
echo "[s2] Computing overlaps..."
uv run python Data/s2_overlap_cases_v1_0.py --article_dir "$ARTICLE_DIR"
echo "[s2] Done: $(date)"

# ----------------------------------------------------------
# Step 3: Build importance_labels.csv
# ----------------------------------------------------------
echo "[s3] Building importance labels..."
uv run python Data/s3_get_importance_v1_1.py --article_dir "$ARTICLE_DIR"
echo "[s3] Done: $(date)"

# ----------------------------------------------------------
# Step 4: Scrape communication-phase texts (subject_matter + questions)
# ----------------------------------------------------------
echo "[s4] Scraping communication-phase texts..."
uv run python Data/s4_extract_text_v1_2.py --article_dir "$ARTICLE_DIR"
echo "[s4] Done: $(date)"

# ----------------------------------------------------------
# Step 5: Scrape judgment outcome texts (fact_section + law_section)
# ----------------------------------------------------------
echo "[s5] Scraping judgment outcome texts..."
uv run python Data/s5_extract_outcome_text_v1_0.py --article_dir "$ARTICLE_DIR"
echo "[s5] Done: $(date)"

echo "========================================"
echo "  Pipeline complete: $(date)"
echo "========================================"
