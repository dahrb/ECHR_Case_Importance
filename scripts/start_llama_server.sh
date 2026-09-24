#!/bin/bash -l
# Serve Llama-3.3-70B-Instruct-FP8 for ECHR_Importance predictions.
# Mirrors scripts/servers/start_llama_server.sh from ADM_JURIX exactly.
# Uses .venv (vLLM 0.19.0 + FA2), NOT .venv_11_2 (GPT-OSS only).
#
#SBATCH -J llama_vllm
#SBATCH -o /users/sgdbareh/scratch/ECHR_Importance/data/data_collection/logs/llama_vllm_%j.out
#SBATCH -e /users/sgdbareh/scratch/ECHR_Importance/data/data_collection/logs/llama_vllm_%j.err
#SBATCH -p gpu-h100
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --mail-user=d.bareham@liverpool.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load cuda/12.8.0-gcc14.2.0

# Llama uses .venv (vLLM 0.19.0 + FA2). NOT .venv_11_2 (that is the GPT-OSS env).
source /users/sgdbareh/scratch/ADM_JURIX/.venv/bin/activate

export HF_HOME="/users/sgdbareh/scratch/ADM_JURIX/LLM_Models/models"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

HF_KEY_PATH="/users/sgdbareh/scratch/ADM_JURIX/LLM_Experiments/hf.key"
if [ -f "$HF_KEY_PATH" ]; then
    export HUGGINGFACE_HUB_TOKEN="$(< "$HF_KEY_PATH")"
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    if [ -d "$NVIDIA_LIB_ROOT" ]; then
        while IFS= read -r -d '' libdir; do
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
        done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
    fi
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PORT=8002
export NODE_NAME=$(hostname)

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
mkdir -p "$REPO_DIR/data/data_collection/logs"

ENDPOINT="http://${NODE_NAME}:${PORT}/v1"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_Llama-3.3-70B_endpoint.txt"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_Llama-3.3-70B-Instruct-FP8_endpoint.txt"
MNT_EP="/mnt/scratch/users/sgdbareh/ECHR_Importance/data/vllm_Llama-3.3-70B-Instruct-FP8_endpoint.txt"
[ -d "$(dirname "$MNT_EP")" ] && echo "$ENDPOINT" > "$MNT_EP" || true

echo "=========================================================="
echo "STARTING vLLM SERVER (Llama-3.3-70B-Instruct-FP8)"
echo "Node:    $NODE_NAME"
echo "URL:     $ENDPOINT"
echo "Job ID:  $SLURM_JOB_ID"
echo "=========================================================="

# FA3 uses Hopper TMA descriptors that crash on PCIe H100 (no NVSwitch); pin FA2.
# --enforce-eager: disables CUDA graph compilation to prevent WorkerProc crash on first inference
# (vLLM 0.19.0 / FP8 / H100-PCIe -- compiled graph crashes in _model_forward on first real batch)
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port $PORT \
    --served-model-name "Llama-3.3-70B-Instruct-FP8" \
    --disable-custom-all-reduce \
    --enable-prefix-caching \
    --enforce-eager \
    --max-model-len 32000 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32000 \
    --attention-config '{"flash_attn_version":2}'
