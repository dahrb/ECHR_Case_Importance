#!/bin/bash
# Materialize the complete reusable retrieval-prompt matrix as one globally
# throttled Slurm array. Submit with:
#   sbatch --array=0-359%24 scripts/materialize_retrieval_matrix.sh
#
# Matrix per article: TGN and FAISS non-RR k=3/5/10, then TGN/FAISS/BM25
# +RR k=3/5/10. Eight deterministic shards per condition gives 45 * 8 = 360 tasks.
#SBATCH --job-name=materialize_prompts
#SBATCH --output=data/data_collection/logs/materialize_prompts_%A_%a.out
#SBATCH --error=data/data_collection/logs/materialize_prompts_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
NUM_SHARDS=8
CONDITIONS_PER_ARTICLE=15

: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false

combo=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
shard=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
article_index=$((combo / CONDITIONS_PER_ARTICLE))
condition=$((combo % CONDITIONS_PER_ARTICLE))
articles=(3 6 8)
article="${articles[$article_index]}"

rerank=0
all_k=0
case "$condition" in
    0|1|2) retriever=tgn_kg; k=$((3 + (condition == 1) * 2 + (condition == 2) * 7)) ;;
    3|4|5) retriever=faiss; offset=$((condition - 3)); k=$((3 + (offset == 1) * 2 + (offset == 2) * 7)) ;;
    6|7|8) retriever=tgn_kg; rerank=1; offset=$((condition - 6)); k=$((3 + (offset == 1) * 2 + (offset == 2) * 7)) ;;
    9) retriever=faiss; rerank=1; all_k=1; k=10 ;;
    10|11) echo "Shared FAISS reranking is handled by condition 9"; exit 0 ;;
    12) retriever=bm25; rerank=1; all_k=1; k=10 ;;
    13|14) echo "Shared BM25 reranking is handled by condition 12"; exit 0 ;;
    *) echo "Invalid condition index: $condition" >&2; exit 1 ;;
esac

args=(
    --article "$article"
    --retriever "$retriever"
    --k "$k"
    --text 1
    --split test
    --max_tokens 1000
    --max_model_len 9600
    --shard_index "$shard"
    --num_shards "$NUM_SHARDS"
    --batch_size 32
    --case_batch_size 64
)

if [ "$rerank" = 1 ]; then
    args+=(
        --rerank
        --rerank_pool 50
        --rerank_model "data/vectordb/article${article}/rerank_model_20260924_ap_selected"
    )
fi
if [ "$all_k" = 1 ]; then
    args+=(--all_k)
fi

echo "article=$article retriever=$retriever k=$k rerank=$rerank shard=$shard/$NUM_SHARDS"
"$VENV/bin/python" echr/prediction/materialize_retrieval_prompts.py "${args[@]}"
