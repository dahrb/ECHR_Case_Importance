#!/bin/bash -l
# Serve the current v2 GPT-OSS LoRA adapters for Articles 3, 6, and 8 on A100.
# A100 needs vLLM MXFP4 dequantization; H100 uses native MXFP4 instead.
#SBATCH --job-name=gptoss_v2_a368_a100
#SBATCH --output=data/data_collection/logs/gptoss_v2_a368_a100_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_v2_a368_a100_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2

set -euo pipefail

MODEL="openai/gpt-oss-120b"
MODEL_NAME="gpt-oss-v2-ft-a100"
PORT="${PORT:-8016}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-48000}"
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/.venv_11_2"
HF_HOME="$ADM_DIR/LLM_Models/models"

cd "$REPO_DIR"
module purge
module load cuda/12.8.0-gcc14.2.0 2>/dev/null || true
source "$VENV/bin/activate"
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" \
       XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True \
       VLLM_MXFP4_DEQUANT=1

if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    if [ -d "$NVIDIA_LIB_ROOT" ]; then
        while IFS= read -r -d '' libdir; do
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
        done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
    fi
fi

HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
if [ -f "$HF_KEY" ]; then
    HF_TOKEN_VALUE="$(< "$HF_KEY")"
    export HF_TOKEN="$HF_TOKEN_VALUE" HUGGINGFACE_HUB_TOKEN="$HF_TOKEN_VALUE"
fi

A3="$REPO_DIR/data/models_v2/gptoss_lora_art3_2ep_noeval/adapter_final"
A6="$REPO_DIR/data/models_v2/gptoss_lora_art6_2ep_noeval/adapter_final"
A8="$REPO_DIR/data/models_v2/gptoss_lora_art8_2ep_fresh_20260926/adapter_final"
for adapter in "$A3" "$A6" "$A8"; do
    [ -f "$adapter/adapter_config.json" ] || { echo "Missing adapter config: $adapter" >&2; exit 1; }
    [ -f "$adapter/adapter_model.safetensors" ] || { echo "Missing adapter weights: $adapter" >&2; exit 1; }
done

NODE=$(hostname)
ENDPOINT="http://${NODE}:${PORT}/v1"
ENDPOINT_FILE="$REPO_DIR/data/vllm_gptoss_v2_a368_a100_endpoint.txt"
rm -f "$ENDPOINT_FILE"
echo "$ENDPOINT" > "$ENDPOINT_FILE"

echo "==== GPT-OSS v2 Art3/Art6/Art8 A100 @ $ENDPOINT (MXFP4 dequant) ===="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora --max-lora-rank 16 --max-loras 3 \
    --lora-modules "gptoss-v2-art3=${A3}" "gptoss-v2-art6=${A6}" "gptoss-v2-art8=${A8}" \
    --max-num-seqs 32
