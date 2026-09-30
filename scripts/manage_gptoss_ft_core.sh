#!/bin/bash
# Manually advance the standalone GPT-OSS FT matrix without Slurm dependencies.
# It follows the base GPT-OSS inference envelope exactly (medium reasoning,
# 8k generation budget, and materialised 16.6k retrieval context); only the
# served model differs.  The job matrix itself is the Llama-style static core:
# zero-shot and plain FAISS only.
#SBATCH --job-name=manage_gptoss_ft_core
#SBATCH --output=data/data_collection/logs/manage_gptoss_ft_core_%j.out
#SBATCH --error=data/data_collection/logs/manage_gptoss_ft_core_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=nodes

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
# Keep the FT endpoint responsive: at most 4 concurrent prediction clients.
MAX_CLIENTS=4
cd "$REPO_DIR"

valid_count() {
    local path="$1"
    [ -f "$path" ] || { echo 0; return; }
    jq -s '[.[] | select(.prediction != null) | .Filename] | unique | length' "$path"
}

target_for() {
    case "$1" in 3) echo 1168 ;; 6) echo 1961 ;; 8) echo 1084 ;; esac
}

output_for() {
    local art="$1" condition="$2" model
    model=$(model_for "$art")
    if [ "$condition" = zero ]; then
        echo "data/results/article${art}/base_text1_test_${model}.jsonl"
    else
        local retriever="${condition%[0-9]*}" k="${condition##*[!0-9]}"
        echo "data/results/article${art}/retrieval_${retriever}_k${k}_text1_test_${model}.jsonl"
    fi
}

active_clients() {
    squeue -u "$(id -un)" -h -o '%j' | awk '/^gft[368]_/{n++} END{print n+0}'
}

endpoint_for() {
    case "$1" in
        3) cat data/vllm_gptoss_merged_art3_native_a100_endpoint.txt 2>/dev/null || true ;;
        8) cat data/vllm_gptoss_merged_art8_native_endpoint.txt 2>/dev/null || true ;;
        6) cat data/vllm_gptoss_merged_art6_native_endpoint.txt 2>/dev/null || true ;;
    esac
}

model_for() {
    case "$1" in
        3) echo gptoss-merged-art3-native-a100 ;;
        8) echo gptoss-merged-art8-native ;;
        6) echo gptoss-merged-art6-native ;;
    esac
}

submit_cell() {
    local art="$1" condition="$2" endpoint="$3" model
    model=$(model_for "$art")
    local common="ALL,ARTICLE=${art},MODEL=${model},ENDPOINT=${endpoint},REASONING_EFFORT=low"
    if [ "$condition" = zero ]; then
        sbatch --parsable --job-name="gft${art}_zero" \
            --export="${common},MAX_TOKENS=4000,MAX_MODEL_LEN=16600" scripts/predict_zero_shot.sh
        return
    fi
    local retriever="${condition%[0-9]*}" k="${condition##*[!0-9]}"
    local prompt="data/prompts/retrieval_v1/article${art}/${retriever}_k${k}_text1_test_ctx16600_out8000"
    local ready
    ready=$(find "$prompt" -maxdepth 1 -type f -name 'part-*-of-0008.jsonl' 2>/dev/null | wc -l)
    [ "$ready" -eq 8 ] || { echo "prompt shards not ready for ${art}/${condition}: ${ready}/8" >&2; return 2; }
    sbatch --parsable --job-name="gft${art}_${condition}" \
        --export="${common},RETRIEVER=${retriever},K=${k},PROMPT_DIR=${prompt},MAX_TOKENS=4000,MAX_MODEL_LEN=16600" \
        scripts/predict_retrieval.sh
}

# Article priority: all Art. 3 cells, then Art. 8, then Art. 6.
cells=(
    '3 zero' '3 faiss3' '3 faiss5' '3 faiss10'
    '8 zero' '8 faiss3' '8 faiss5' '8 faiss10'
    '6 zero' '6 faiss3' '6 faiss5' '6 faiss10'
)

while :; do
    active=$(active_clients)
    pending=0
    for cell in "${cells[@]}"; do
        read -r art condition <<< "$cell"
        path=$(output_for "$art" "$condition")
        target=$(target_for "$art")
        count=$(valid_count "$path")
        if [ "$count" -ge "$target" ]; then
            continue
        fi
        endpoint=$(endpoint_for "$art")
        if [ -z "$endpoint" ] || ! curl -sf --max-time 8 "${endpoint%/v1}/health" >/dev/null; then
            echo "$(date --iso-8601=seconds) art${art} endpoint unavailable; waiting"
            continue
        fi
        pending=1
        if [ "$active" -lt "$MAX_CLIENTS" ]; then
            job=$(submit_cell "$art" "$condition" "$endpoint" || true)
            [ -n "$job" ] || continue
            echo "$(date --iso-8601=seconds) submitted ${art}/${condition}: ${job} (${count}/${target})"
            active=$((active + 1))
        fi
    done
    [ "$pending" -eq 0 ] && { echo "GPT-OSS FT core matrix complete"; exit 0; }
    sleep 60
done
