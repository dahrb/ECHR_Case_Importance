#!/bin/bash
# Start a vLLM server for ECHR_Importance predictions.
# Uses the ADM_JURIX .venv_gptoss environment and HF model cache.
#
# Default model: openai/gpt-oss-20b  (set MODEL env var to override)
# Default port:  8000               (set PORT env var to override)
#
# Usage:
#   sbatch scripts/start_vllm_server.sh
#   sbatch --export=MODEL=nvidia/Llama-3.3-70B-Instruct-FP8,MODEL_NAME=Llama-3.3-70B scripts/start_vllm_server.sh
#
# The running node's endpoint is written to data/vllm_endpoint.txt so that
# prediction/summarization jobs can find the server.
#
#SBATCH --job-name=vllm_server
#SBATCH --output=data/data_collection/logs/vllm_server_%j.out
#SBATCH --error=data/data_collection/logs/vllm_server_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-h100

set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-20b}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-20b}"
PORT="${PORT:-8000}"
TP="${TP:-2}"   # tensor parallel size
VENV_NAME="${VENV_NAME:-.venv_11_2}"   # .venv for Llama, .venv_11_2 for GPT-OSS
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/$VENV_NAME"
HF_HOME="$ADM_DIR/LLM_Models/models"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

module purge
module load cuda/12.8.0-gcc14.2.0 2>/dev/null || true

source "$VENV/bin/activate"

# expose pip-installed CUDA runtime libs
if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    if [ -d "$NVIDIA_LIB_ROOT" ]; then
        while IFS= read -r -d '' libdir; do
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
        done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
    fi
fi

export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True

NODE_NAME=$(hostname)
ENDPOINT="http://${NODE_NAME}:${PORT}/v1"
# Write to model-specific endpoint file; also update the shared file only for gpt-oss
SAFE_NAME="${MODEL_NAME//\//_}"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_${SAFE_NAME}_endpoint.txt"
if [[ "$MODEL_NAME" == gpt-oss* ]]; then
    echo "$ENDPOINT" > "$REPO_DIR/data/vllm_endpoint.txt"
fi

echo "========================================"
echo "  vLLM Server — ECHR_Importance"
echo "  Model:    $MODEL (served as $MODEL_NAME)"
echo "  Endpoint: $ENDPOINT"
echo "  TP size:  $TP"
echo "  Started:  $(date)"
echo "========================================"

vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    --max-num-seqs 32
