#!/bin/bash
# Materialize only the GPT-OSS FT core retrieval prompts: Articles 3/6/8,
# plain FAISS and TGN k=3/5/10, eight deterministic shards per cell.
# 16,600 context minus 8,000 output minus 768 overhead preserves the existing
# 7,832-token retrieval-input budget from ctx9600_out1000 exactly.
#SBATCH --job-name=materialize_gptft
#SBATCH --output=data/data_collection/logs/materialize_gptft_%A_%a.out
#SBATCH --error=data/data_collection/logs/materialize_gptft_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit as --array=0-143%24}"
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" TOKENIZERS_PARALLELISM=false

shard=$((SLURM_ARRAY_TASK_ID % 8))
cell=$((SLURM_ARRAY_TASK_ID / 8))
article_index=$((cell / 6))
condition=$((cell % 6))
articles=(3 6 8)
article="${articles[$article_index]}"
case "$condition" in
  0) retriever=faiss; k=3 ;;
  1) retriever=faiss; k=5 ;;
  2) retriever=faiss; k=10 ;;
  3) retriever=tgn_kg; k=3 ;;
  4) retriever=tgn_kg; k=5 ;;
  5) retriever=tgn_kg; k=10 ;;
esac

echo "article=$article retriever=$retriever k=$k shard=$shard/8"
"$VENV/bin/python" echr/prediction/materialize_retrieval_prompts.py \
  --article "$article" --retriever "$retriever" --k "$k" --text 1 --split test \
  --max_tokens 8000 --max_model_len 16600 --shard_index "$shard" --num_shards 8 \
  --batch_size 32 --case_batch_size 64
