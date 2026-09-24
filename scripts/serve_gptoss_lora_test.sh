#!/bin/bash -l
# TEST: serve openai/gpt-oss-120b + a trained LoRA adapter via vLLM on A100.
# Verifies the runtime-LoRA serving path (no merge) works for the fine-tuned adapters.
#
# Uses a DISTINCT served-model-name + endpoint file so it does NOT clobber the
# running H100 prediction server (data/vllm_endpoint.txt / vllm_gptoss_endpoint.txt).
#
# Usage:
#   sbatch --export=ADAPTER=data/models/gptoss_lora_art3/adapter_final,LORA_NAME=echr_art3 \
#          scripts/serve_gptoss_lora_test.sh
#
#SBATCH --job-name=gptoss_lora_test
#SBATCH --output=data/data_collection/logs/gptoss_lora_test_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_lora_test_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:4

set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-120b}"
MODEL_NAME="gpt-oss-120b-ft-test"          # distinct → separate endpoint file
PORT="${PORT:-8010}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
ADAPTER="${ADAPTER:?Must set ADAPTER=path/to/adapter_final}"
LORA_NAME="${LORA_NAME:-echr_ft}"
VENV_NAME="${VENV_NAME:-.venv_11_2}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/$VENV_NAME"
HF_HOME="$ADM_DIR/LLM_Models/models"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs

module purge
module load cuda/12.8.0-gcc14.2.0 2>/dev/null || true
source "$VENV/bin/activate"

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
# gpt-oss on A100 (no Hopper MXFP4 kernels) → force bf16 dequant path
export VLLM_MXFP4_DEQUANT=1

HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }

ABS_ADAPTER="$REPO_DIR/$ADAPTER"
[ -d "$ABS_ADAPTER" ] || { echo "ERROR: adapter not found: $ABS_ADAPTER" >&2; exit 1; }

NODE_NAME=$(hostname)
ENDPOINT="http://${NODE_NAME}:${PORT}/v1"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_gptoss_lora_test_endpoint.txt"

echo "========================================"
echo "  vLLM LoRA TEST — GPT-OSS + adapter"
echo "  Base:     $MODEL (served as $MODEL_NAME)"
echo "  Adapter:  $LORA_NAME = $ABS_ADAPTER"
echo "  Endpoint: $ENDPOINT   TP=$TP"
echo "  Started:  $(date)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${ABS_ADAPTER}" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-num-seqs 16
