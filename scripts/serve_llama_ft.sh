#!/bin/bash -l
# Serve meta-llama/Llama-3.3-70B-Instruct (bf16) + the 3 per-article FT LoRA
# adapters (art3/6/8) via vLLM --enable-lora on 2xA100. For the FT prediction
# condition (zero-shot base prompt on the fine-tuned model). Distinct endpoint file.
#
#SBATCH --job-name=llama_ft_serve
#SBATCH --output=data/data_collection/logs/llama_ft_serve_%j.out
#SBATCH --error=data/data_collection/logs/llama_ft_serve_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2

set -euo pipefail

MODEL="meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME="Llama-3.3-70B-ft"
PORT="${PORT:-8012}"
TP=2
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16000}"   # bf16 70B on 2xA100 leaves limited KV cache; 16k covers our prompts
VENV_NAME=".venv_11_2"

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

A3="$REPO_DIR/data/models/llama_lora_art3/adapter_final"
A6="$REPO_DIR/data/models/llama_lora_art6/adapter_final"
A8="$REPO_DIR/data/models/llama_lora_art8/adapter_final"

NODE=$(hostname); ENDPOINT="http://${NODE}:${PORT}/v1"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_llama_ft_endpoint.txt"
echo "==== Llama FT serve @ $ENDPOINT (adapters: ft-art3/6/8) ===="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" --gpu-memory-utilization 0.95 \
    --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL_NAME" \
    --trust-remote-code --max-model-len "$MAX_MODEL_LEN" \
    --enable-lora --max-lora-rank 16 --max-loras 3 \
    --lora-modules "llama-ft-art3=${A3}" "llama-ft-art6=${A6}" "llama-ft-art8=${A8}" \
    --max-num-seqs 32
