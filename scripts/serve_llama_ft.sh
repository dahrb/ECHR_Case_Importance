#!/bin/bash -l
# Serve the BF16 training base plus both epoch checkpoints for each new adapter.
#
#SBATCH --job-name=llama_ft_serve
#SBATCH --output=data/data_collection/logs/llama_ft_serve_%j.out
#SBATCH --error=data/data_collection/logs/llama_ft_serve_%j.err
#SBATCH --partition=gpu-a100-lowbig
# Validation only: loading plus three small held-out runs fits comfortably here,
# and the short request greatly improves A100 backfill scheduling.
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --no-requeue

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
MODEL_NAME="Llama-3.3-70B-ft-v2"
PORT="${PORT:-8012}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
VENV_NAME="${VENV_NAME:-.venv_11_2}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/$VENV_NAME"
HF_HOME="$ADM_DIR/LLM_Models/models"

cd "$REPO_DIR"; mkdir -p data/data_collection/logs
module purge; module load cuda/12.8.0-gcc14.2.0 2>/dev/null || true
source "$VENV/bin/activate"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    [ -d "$NVIDIA_LIB_ROOT" ] && while IFS= read -r -d '' d; do export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
fi
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"; [ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }

A3="$REPO_DIR/data/models_v2/llama_lora_art3_2ep_1gpu"
A6="$REPO_DIR/data/models_v2/llama_lora_art6_2ep_1gpu"
A8="$REPO_DIR/data/models_v2/llama_lora_art8_2ep_1gpu"
for adapter in "$A3/checkpoint-21" "$A3/checkpoint-42" \
               "$A6/checkpoint-62" "$A6/checkpoint-124" \
               "$A8/checkpoint-34" "$A8/checkpoint-68"; do
    [ -f "$adapter/adapter_model.safetensors" ] || { echo "Missing $adapter" >&2; exit 1; }
done

NODE=$(hostname); ENDPOINT="http://${NODE}:${PORT}/v1"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_llama_ft_endpoint.txt"
echo "==== Llama FT validation serve @ $ENDPOINT (model=$MODEL, epoch 1/2 adapters) ===="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_ARGS=()
if [[ "$MODEL" == *FP8* ]]; then
    # NVIDIA ModelOpt FP8 requires Hopper (compute capability >= 8.9).
    MODEL_ARGS+=(--enforce-eager)
fi

vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL_NAME" \
    --trust-remote-code --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora --max-lora-rank 16 --max-loras 1 --max-cpu-loras 6 \
    --lora-modules \
      "llama-v2-art3-e1=$A3/checkpoint-21" "llama-v2-art3-e2=$A3/checkpoint-42" \
      "llama-v2-art6-e1=$A6/checkpoint-62" "llama-v2-art6-e2=$A6/checkpoint-124" \
      "llama-v2-art8-e1=$A8/checkpoint-34" "llama-v2-art8-e2=$A8/checkpoint-68" \
    --max-num-seqs 16 "${MODEL_ARGS[@]}"
