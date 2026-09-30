#!/bin/bash
# Materialize iterative zero-shot prompts for Articles 3/6/8 in eight shards.
# Submit with: sbatch --array=0-23%12 scripts/materialize_gptoss_iterative_zero_prompts.sh
#SBATCH --job-name=materialize_gptiter0
#SBATCH --output=data/data_collection/logs/materialize_gptiter0_%A_%a.out
#SBATCH --error=data/data_collection/logs/materialize_gptiter0_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=nodes

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit as --array=0-23%12}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

shard=$((SLURM_ARRAY_TASK_ID % 8))
article_index=$((SLURM_ARRAY_TASK_ID / 8))
articles=(3 6 8)
article="${articles[$article_index]}"

"$VENV/bin/python" echr/prediction/materialize_iterative_zero_prompts.py \
  --article "$article" --text 1 --split test --max_tokens "$MAX_TOKENS" --max_model_len "$MAX_MODEL_LEN" \
  --shard_index "$shard" --num_shards 8
