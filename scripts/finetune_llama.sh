#!/bin/bash -l
# QLoRA Fine-tuning: meta-llama/Llama-3.3-70B-Instruct (bf16, gated — needs hf.key)
# Parameterized by TAG (art3 | art6 | art8). Combined is intentionally excluded.
#
# Usage:
#   sbatch --export=TAG=art3 scripts/finetune_llama.sh          # Art3 adapter (first)
#   sbatch --export=TAG=art6,PARTITION=gpu-h100 scripts/finetune_llama.sh
#
# Default partition: gpu-a100-lowbig (4x A100 80GB nodes, leaves H100 for inference).
#
#SBATCH --job-name=ft_llama_echr
#SBATCH --output=data/data_collection/logs/ft_llama_%j.out
#SBATCH --error=data/data_collection/logs/ft_llama_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --no-requeue

set -euo pipefail

TAG="${TAG:-}"
case "$TAG" in art3|art6|art8) ;; *) echo "ERROR: TAG must be art3, art6, or art8" >&2; exit 2 ;; esac
# Base model + adapter-dir suffix are overridable so we can retrain on FP8:
#   sbatch --export=TAG=art3,MODEL_ID=nvidia/Llama-3.3-70B-Instruct-FP8,ADAPTER_SUFFIX=_fp8 scripts/finetune_llama.sh
# FP8 base auto-selects --load_mode asis (LoRA on frozen FP8 base); bnb 4-bit cannot
# re-quantize float8 (see gotcha memory). bf16 default reproduces the original run.
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"
ADAPTER_SUFFIX="${ADAPTER_SUFFIX:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

module purge
module load cuda/12.8.0-gcc14.2.0

WORK_DIR="/users/sgdbareh/scratch/ADM_JURIX"
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="$WORK_DIR/.venv"   # has peft, trl, bitsandbytes

source "$VENV/bin/activate"

hostname && echo "Started: $(date)  TAG=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO_DIR"

DATA_DIR="data/finetune_v2/$TAG"
for f in "$DATA_DIR/sft_train.jsonl" "$DATA_DIR/sft_val.jsonl"; do
    if [ ! -f "$REPO_DIR/$f" ]; then
        echo "ERROR: $f not found. Run scripts/create_finetune_data.sh first." >&2
        exit 1
    fi
done
echo "Train: $(wc -l < $DATA_DIR/sft_train.jsonl) examples"
echo "Val:   $(wc -l < $DATA_DIR/sft_val.jsonl) examples"

ADAPTER_DIR="$REPO_DIR/data/models_v2/llama_lora_${TAG}_2ep${ADAPTER_SUFFIX}"
if [ -e "$ADAPTER_DIR" ] && [ -z "$RESUME_FROM_CHECKPOINT" ]; then
    echo "ERROR: fresh-run output already exists: $ADAPTER_DIR" >&2
    exit 3
fi
mkdir -p "$ADAPTER_DIR" data/data_collection/logs

export HF_HOME="$WORK_DIR/LLM_Models/models"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$HF_HOME/xdg_cache"

HF_KEY="$WORK_DIR/LLM_Experiments/hf.key"
if [ -f "$HF_KEY" ]; then
    export HUGGINGFACE_HUB_TOKEN="$(< "$HF_KEY")"
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi

# NVIDIA lib paths (same pattern as ADM_JURIX)
if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    if [ -d "$NVIDIA_LIB_ROOT" ]; then
        while IFS= read -r -d '' libdir; do
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
        done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
    fi
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== QLoRA Fine-tuning: $MODEL_ID  (TAG=$TAG → adapter llama_lora_${TAG}${ADAPTER_SUFFIX}) ==="

python echr/finetune/finetune_echr.py \
    --model_id "$MODEL_ID" \
    --train_dataset "$DATA_DIR/sft_train.jsonl" \
    --val_dataset   "$DATA_DIR/sft_val.jsonl" \
    --output_dir    "$ADAPTER_DIR" \
    --epochs 2 \
    --batch_size 1 \
    --grad_accum 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_len "${MAX_SEQ_LEN:-4096}" \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}

echo ""
echo "Done: $(date). Adapter → $ADAPTER_DIR/adapter_final"
