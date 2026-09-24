#!/bin/bash -l
# Serve a merged GPT-OSS-120B model on H100 (native MXFP4, no dequant).
# Parallel alternative to serve_gptoss_merged.sh (A100 version).
# Writes to same endpoint file → prediction jobs use whichever starts first.
#
# MXFP4 format: TP=2 (native H100 support, no VLLM_MXFP4_DEQUANT)
# bf16 format:  TP=4 (4×80GB H100 = 320GB, fits 240GB model)
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/serve_gptoss_merged_h100.sh
#
#SBATCH --job-name=gptoss_merged_h100
#SBATCH --output=data/data_collection/logs/gptoss_merged_h100_art%a_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_merged_h100_art%a_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

PORT="${PORT:-8014}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-48000}"
VENV_NAME=".venv_11_2"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/$VENV_NAME"
HF_HOME="$ADM_DIR/LLM_Models/models"
MODEL_DIR="$REPO_DIR/data/models/gptoss_merged_art${ARTICLE}"
ENDPOINT_FILE="$REPO_DIR/data/vllm_gptoss_merged_art${ARTICLE}_endpoint.txt"
MNT_ENDPOINT_FILE="/mnt/scratch/users/sgdbareh/ECHR_Importance/data/vllm_gptoss_merged_art${ARTICLE}_endpoint.txt"

cd "$REPO_DIR"; mkdir -p data/data_collection/logs

module purge; module load cuda/12.8.0-gcc14.2.0 2>/dev/null || true
source "$VENV/bin/activate"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    [ -d "$NVIDIA_LIB_ROOT" ] && while IFS= read -r -d '' d; do
        export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
fi
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
# H100: native MXFP4 — do NOT set VLLM_MXFP4_DEQUANT
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }

FORMAT_FILE="$MODEL_DIR/FORMAT"
if [ ! -f "$FORMAT_FILE" ]; then
    echo "ERROR: $FORMAT_FILE not found — merge job must complete first." >&2
    exit 1
fi
FORMAT=$(cat "$FORMAT_FILE")
echo "Model format: $FORMAT"

if [ "$FORMAT" = "mxfp4" ]; then
    TP=2
    echo "Serving MXFP4 model natively on H100 with TP=$TP"
else
    TP=4
    echo "Serving bf16 model with TP=$TP"
fi

MODEL_NAME="gptoss-ft-art${ARTICLE}"
NODE=$(hostname)
ENDPOINT="http://${NODE}:${PORT}/v1"

rm -f "$ENDPOINT_FILE" "$MNT_ENDPOINT_FILE"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
[ -d "$(dirname "$MNT_ENDPOINT_FILE")" ] && echo "$ENDPOINT" > "$MNT_ENDPOINT_FILE" || true

echo "==== GPT-OSS merged art${ARTICLE} serve @ $ENDPOINT (H100, TP=$TP, format=$FORMAT) ===="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

vllm serve "$MODEL_DIR" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 16
