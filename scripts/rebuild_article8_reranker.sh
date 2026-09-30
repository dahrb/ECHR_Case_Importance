#!/bin/bash
# AP-selected, group-disjoint LegalBERT re-ranker rebuild.
#SBATCH --job-name=rerank_rebuild
#SBATCH --output=data/data_collection/logs/rerank_rebuild_%j.out
#SBATCH --error=data/data_collection/logs/rerank_rebuild_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s-low

set -euo pipefail
: "${ARTICLE:?Must set ARTICLE (3, 6, or 8)}"
cd /users/sgdbareh/scratch/ECHR_Importance
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/mnt/data1/users/sgdbareh/venvs/ECHR_Importance/bin/python \
  echr/rerank/rebuild_article8_reranker.py --article "$ARTICLE" --epochs 15 --min-ap 0.85
