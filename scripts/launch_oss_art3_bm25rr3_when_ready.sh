#!/bin/bash
# Poll independently for the final Article 3 BM25+RR k=3 prompt shard and
# submit GPT-OSS inference immediately after the complete prompt set validates.
# This deliberately uses no Slurm dependency.
#SBATCH --job-name=gate_oss_a3_brr3
#SBATCH --output=data/data_collection/logs/gate_oss_a3_brr3_%j.out
#SBATCH --error=data/data_collection/logs/gate_oss_a3_brr3_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=nodes

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
PROMPT_DIR="data/prompts/retrieval_v1/article3/bm25_k3_rerank-rerank_model_20260924_ap_selected_text1_test_ctx9600_out1000"
ENDPOINT="http://gpu31.barkla2.liv.alces.network:8013/v1"
TARGET_ROWS=1168

cd "$REPO_DIR"

while true; do
    shard_count=$(find "$PROMPT_DIR" -maxdepth 1 -type f -name 'part-*.jsonl' | wc -l)
    row_count=0
    if [ "$shard_count" -eq 8 ]; then
        row_count=$(find "$PROMPT_DIR" -maxdepth 1 -type f -name 'part-*.jsonl' -print0 \
            | xargs -0 wc -l | awk 'END {print ($2 == "total" ? $1 : $1 + 0)}')
    fi

    printf '%s shards=%s/8 rows=%s/%s\n' "$(date --iso-8601=seconds)" \
        "$shard_count" "$row_count" "$TARGET_ROWS"

    if [ "$shard_count" -eq 8 ] && [ "$row_count" -eq "$TARGET_ROWS" ]; then
        if squeue -h -u "$USER" -n oss_a3_brr3 | grep -q .; then
            echo "oss_a3_brr3 is already queued or running; exiting"
            exit 0
        fi

        output="data/results/article3/retrieval_bm25_k3_rerank_text1_test_gpt-oss-120b_static_rerun_20260925.jsonl"
        if [ -f "$output" ]; then
            valid=$(jq -s '[.[] | select(.prediction != null) | .Filename] | unique | length' "$output")
            if [ "$valid" -ge "$TARGET_ROWS" ]; then
                echo "Article 3 BM25+RR k=3 is already complete; exiting"
                exit 0
            fi
        fi

        until curl -sf --max-time 5 "${ENDPOINT%/v1}/health" >/dev/null; do
            echo "GPT-OSS endpoint is not healthy; retrying in 30 seconds"
            sleep 30
        done

        job_id=$(sbatch --parsable --job-name=oss_a3_brr3 \
            --export="ALL,ARTICLE=3,MODEL=gpt-oss-120b,RETRIEVER=bm25,K=3,RERANK=1,RERANK_MODEL=data/vectordb/article3/rerank_model_20260924_ap_selected,RERANK_POOL=50,PROMPT_DIR=${PROMPT_DIR},RUN_TAG=static_rerun_20260925,ENDPOINT=${ENDPOINT},MAX_TOKENS=1000,MAX_MODEL_LEN=9600" \
            scripts/predict_retrieval.sh)
        echo "Submitted oss_a3_brr3 as job $job_id"
        exit 0
    fi

    sleep 30
done
