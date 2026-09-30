#!/bin/bash -l
# Serve GPT-OSS-120B with all three article LoRA adapters on H100s.
# H100 uses native MXFP4 kernels, so no MXFP4 dequantization flag is set.
#SBATCH --job-name=gptoss_ft_h100
#SBATCH --output=data/data_collection/logs/gptoss_ft_h100_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_ft_h100_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2

set -euo pipefail

MODEL="openai/gpt-oss-120b"
MODEL_NAME="gpt-oss-120b-ft-h100"
PORT="${PORT:-8015}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-48000}"
VENV_NAME=".venv_11_2"
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
    [ -d "$NVIDIA_LIB_ROOT" ] && while IFS= read -r -d '' d; do
        export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
fi
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" \
       XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }

A3="$REPO_DIR/data/models/gptoss_lora_art3/adapter_final"
A6="$REPO_DIR/data/models/gptoss_lora_art6/adapter_final"
A8="$REPO_DIR/data/models/gptoss_lora_art8/adapter_final"
for adapter in "$A3" "$A6" "$A8"; do
    [ -d "$adapter" ] || { echo "Missing LoRA adapter: $adapter" >&2; exit 1; }
done

NODE=$(hostname)
ENDPOINT="http://${NODE}:${PORT}/v1"
ENDPOINT_FILE="$REPO_DIR/data/vllm_gptoss_ft_h100_endpoint.txt"
MNT_ENDPOINT_FILE="/mnt/scratch/users/sgdbareh/ECHR_Importance/data/vllm_gptoss_ft_h100_endpoint.txt"
rm -f "$ENDPOINT_FILE" "$MNT_ENDPOINT_FILE"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
[ -d "$(dirname "$MNT_ENDPOINT_FILE")" ] && echo "$ENDPOINT" > "$MNT_ENDPOINT_FILE" || true

echo "==== GPT-OSS FT H100 @ $ENDPOINT (TP=2, native MXFP4) ===="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora --max-lora-rank 16 --max-loras 3 \
    --lora-modules "gptoss-ft-art3=${A3}" "gptoss-ft-art6=${A6}" "gptoss-ft-art8=${A8}" \
    --max-num-seqs 32
