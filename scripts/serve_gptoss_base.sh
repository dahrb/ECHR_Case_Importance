#!/bin/bash -l
# Serve base openai/gpt-oss-120b on H100s for reasoning-trace generation.
# Does NOT load any LoRA adapter — plain base model only.
# Writes endpoint to data/vllm_gptoss_base_endpoint.txt (distinct from FT endpoint).
#
#SBATCH --job-name=gptoss_base_serve
#SBATCH --output=data/data_collection/logs/gptoss_base_serve_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_base_serve_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2

set -euo pipefail

MODEL="openai/gpt-oss-120b"
MODEL_NAME="${MODEL_NAME:-gpt-oss-120b}"
PORT="${PORT:-8013}"
TP=2
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
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
    [ -d "$NVIDIA_LIB_ROOT" ] && while IFS= read -r -d '' d; do
        export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
fi
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" \
       XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }

# Remove stale endpoint file before writing so gen_traces.py doesn't connect to
# a dead node from a previous run.
rm -f "$REPO_DIR/data/vllm_gptoss_base_endpoint.txt"
NODE=$(hostname)
ENDPOINT="http://${NODE}:${PORT}/v1"
echo "$ENDPOINT" > "$REPO_DIR/data/vllm_gptoss_base_endpoint.txt"
# Also write to mnt path in case /users symlink differs on compute nodes
MNT_EP="/mnt/scratch/users/sgdbareh/ECHR_Importance/data/vllm_gptoss_base_endpoint.txt"
[ -d "$(dirname "$MNT_EP")" ] && echo "$ENDPOINT" > "$MNT_EP" || true
echo "==== GPT-OSS base serve @ $ENDPOINT (TP=$TP, no LoRA) ===="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 32
