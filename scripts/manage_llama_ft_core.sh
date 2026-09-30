#!/bin/bash
# Independently advance the selected Llama FT core matrix without Slurm
# dependencies. Epoch-1 was selected for Articles 6 and 8 after a complete
# held-out tie with epoch 2; epoch 1 preserves their valid provisional TGN runs.
# Stages use at most three concurrent clients for the same LoRA adapter.
#SBATCH --job-name=manage_llama_ft_core
#SBATCH --output=data/data_collection/logs/manage_llama_ft_core_%j.out
#SBATCH --error=data/data_collection/logs/manage_llama_ft_core_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=nodes

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ENDPOINT="http://gpu09.barkla2.liv.alces.network:8012/v1"
RUN_TAG="static_rerun_20260925"

cd "$REPO_DIR"

valid_count() {
    local file="$1"
    [ -f "$file" ] || { echo 0; return; }
    jq -s '[.[] | select(.prediction != null) | .Filename] | unique | length' "$file"
}

wait_complete() {
    local file="$1" target="$2" label="$3"
    while true; do
        local n
        n=$(valid_count "$file")
        printf '%s %s %s/%s\n' "$(date --iso-8601=seconds)" "$label" "$n" "$target"
        [ "$n" -ge "$target" ] && return
        sleep 60
    done
}

submit_zero() {
    local art="$1" model="$2" name="$3"
    sbatch --parsable --job-name="$name" \
        --export="ALL,ARTICLE=${art},MODEL=${model},ENDPOINT=${ENDPOINT},MAX_TOKENS=1000,MAX_MODEL_LEN=48000" \
        scripts/predict_zero_shot.sh
}

submit_faiss() {
    local art="$1" model="$2" k="$3" name="$4"
    local prompt="data/prompts/retrieval_v1/article${art}/faiss_k${k}_text1_test_ctx9600_out1000"
    sbatch --parsable --job-name="$name" \
        --export="ALL,ARTICLE=${art},MODEL=${model},RETRIEVER=faiss,K=${k},PROMPT_DIR=${prompt},RUN_TAG=${RUN_TAG},ENDPOINT=${ENDPOINT},MAX_TOKENS=1000,MAX_MODEL_LEN=9600" \
        scripts/predict_retrieval.sh
}

submit_tgn() {
    local art="$1" model="$2" k="$3" name="$4"
    local prompt="data/prompts/retrieval_v1/article${art}/tgn_kg_k${k}_text1_test_ctx9600_out1000"
    sbatch --parsable --job-name="$name" \
        --export="ALL,ARTICLE=${art},MODEL=${model},RETRIEVER=tgn_kg,K=${k},PROMPT_DIR=${prompt},RUN_TAG=${RUN_TAG},ENDPOINT=${ENDPOINT},MAX_TOKENS=1000,MAX_MODEL_LEN=9600" \
        scripts/predict_retrieval.sh
}

# Art. 6 zero-shot / FAISS k=3/5 were submitted before this manager.
wait_complete data/results/article6/base_text1_test_llama-v2-art6-e1.jsonl 1961 art6_zero
wait_complete data/results/article6/retrieval_faiss_k3_text1_test_llama-v2-art6-e1_static_rerun_20260925.jsonl 1961 art6_faiss3
wait_complete data/results/article6/retrieval_faiss_k5_text1_test_llama-v2-art6-e1_static_rerun_20260925.jsonl 1961 art6_faiss5

submit_faiss 6 llama-v2-art6-e1 10 llft6_faiss10
wait_complete data/results/article6/retrieval_faiss_k10_text1_test_llama-v2-art6-e1_static_rerun_20260925.jsonl 1961 art6_faiss10

# Then switch the single active adapter to Article 8 epoch 1.
submit_zero 8 llama-v2-art8-e1 llft8_zero
submit_faiss 8 llama-v2-art8-e1 3 llft8_faiss3
submit_faiss 8 llama-v2-art8-e1 5 llft8_faiss5
wait_complete data/results/article8/base_text1_test_llama-v2-art8-e1.jsonl 1084 art8_zero
wait_complete data/results/article8/retrieval_faiss_k3_text1_test_llama-v2-art8-e1_static_rerun_20260925.jsonl 1084 art8_faiss3
wait_complete data/results/article8/retrieval_faiss_k5_text1_test_llama-v2-art8-e1_static_rerun_20260925.jsonl 1084 art8_faiss5

# Finish the Article 8 core matrix, including its two incomplete provisional
# TGN files. TGN k=3 already has full valid coverage.
submit_faiss 8 llama-v2-art8-e1 10 llft8_faiss10
submit_tgn 8 llama-v2-art8-e1 5 llft8_tgn5_repair
submit_tgn 8 llama-v2-art8-e1 10 llft8_tgn10_repair
wait_complete data/results/article8/retrieval_faiss_k10_text1_test_llama-v2-art8-e1_static_rerun_20260925.jsonl 1084 art8_faiss10
wait_complete data/results/article8/retrieval_tgn_kg_k5_text1_test_llama-v2-art8-e1_static_rerun_20260925.jsonl 1084 art8_tgn5
wait_complete data/results/article8/retrieval_tgn_kg_k10_text1_test_llama-v2-art8-e1_static_rerun_20260925.jsonl 1084 art8_tgn10

echo "Core Llama FT matrix complete: $(date --iso-8601=seconds)"
