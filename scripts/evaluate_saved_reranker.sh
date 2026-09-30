#!/bin/bash
# Evaluate existing Art. 3/6 rerankers only; no training or model writes.
#SBATCH --job-name=eval_reranker
#SBATCH --output=data/data_collection/logs/eval_reranker_%j.out
#SBATCH --error=data/data_collection/logs/eval_reranker_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-l40s-low

set -euo pipefail
: "${ARTICLE:?}" "${FOLD:?}" "${MODEL_DIR:?}"
cd /users/sgdbareh/scratch/ECHR_Importance
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/mnt/data1/users/sgdbareh/venvs/ECHR_Importance/bin/python \
  echr/rerank/evaluate_saved_reranker.py \
  --article "$ARTICLE" --fold "$FOLD" --model-dir "$MODEL_DIR"
