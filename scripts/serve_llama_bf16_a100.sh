#!/bin/bash -l
# A100 fallback for Llama when FP8 H100 capacity is unavailable.
# This intentionally uses bf16 base weights, not the FP8 ModelOpt checkpoint.
#SBATCH -J llama_bf16_vllm
#SBATCH -o data/data_collection/logs/llama_bf16_server_%j.out
#SBATCH -e data/data_collection/logs/llama_bf16_server_%j.err
#SBATCH -p gpu-a100-lowbig
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2

set -euo pipefail

module purge
module load cuda/12.8.0-gcc14.2.0
source /users/sgdbareh/scratch/ADM_JURIX/.venv/bin/activate

export HF_HOME="/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -f /users/sgdbareh/scratch/ADM_JURIX/LLM_Experiments/hf.key ]; then
    export HF_TOKEN="$(< /users/sgdbareh/scratch/ADM_JURIX/LLM_Experiments/hf.key)"
fi

PORT="${PORT:-8000}"
NODE_NAME=$(hostname)
ENDPOINT_FILE="${ENDPOINT_FILE:?Must provide ENDPOINT_FILE}"
# Two 80 GB A100s can load the BF16 70B model, but not retain a 32k-token
# KV cache.  8k is below the measured 9,920-token ceiling from job 10689457.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
mkdir -p "$(dirname "$ENDPOINT_FILE")"
endpoint_tmp="${ENDPOINT_FILE}.tmp.${SLURM_JOB_ID}"
printf 'http://%s:%s/v1\n' "$NODE_NAME" "$PORT" > "$endpoint_tmp"
mv "$endpoint_tmp" "$ENDPOINT_FILE"
trap 'rm -f "$ENDPOINT_FILE"' EXIT

echo "Llama BF16 A100 server: http://$NODE_NAME:$PORT/v1"
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "Llama-3.3-70B-Instruct" \
    --disable-custom-all-reduce \
    --enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --attention-config '{"flash_attn_version":2}'
