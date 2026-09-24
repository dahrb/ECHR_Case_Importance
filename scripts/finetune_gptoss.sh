#!/bin/bash -l
# QLoRA Fine-tuning: openai/gpt-oss-120b
# Parameterized by TAG (art3 | art6 | art8 | combined).
# Run AFTER the Llama adapters (user requested Llama first).
# Requires: data/finetune/$TAG/sft_train.jsonl + sft_val.jsonl to exist.
#
# Usage:
#   sbatch --export=TAG=art3 scripts/finetune_gptoss.sh
#   sbatch --export=TAG=combined scripts/finetune_gptoss.sh
#
# Default partition: gpu-a100-lowbig.
#
#SBATCH --job-name=ft_gptoss_echr
#SBATCH --output=data/data_collection/logs/ft_gptoss_%j.out
#SBATCH --error=data/data_collection/logs/ft_gptoss_%j.err
# gpt-oss-120b dequantized (MXFP4→bf16) ≈ 240GB → needs 4× A100 80GB.
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4

set -euo pipefail

TAG="${TAG:-combined}"
# Set HARMONY_SFT=1 to use reasoning-trace data (traces_train/val.jsonl) and
# the harmony-channel SFT mode (analysis + final channels in training sequences).
HARMONY_SFT="${HARMONY_SFT:-0}"

module purge
module load cuda/12.8.0-gcc14.2.0

WORK_DIR="/users/sgdbareh/scratch/ADM_JURIX"
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="$WORK_DIR/.venv"

source "$VENV/bin/activate"

hostname && echo "Started: $(date)  TAG=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO_DIR"

DATA_DIR="data/finetune/$TAG"
if [ "$HARMONY_SFT" = "1" ]; then
    TRAIN_FILE="$DATA_DIR/traces_train.jsonl"
    VAL_FILE="$DATA_DIR/traces_val.jsonl"
    HARMONY_FLAG="--harmony_sft"
    echo "Mode: HARMONY SFT (reasoning traces + analysis+final channels)"
else
    TRAIN_FILE="$DATA_DIR/sft_train.jsonl"
    VAL_FILE="$DATA_DIR/sft_val.jsonl"
    HARMONY_FLAG=""
    echo "Mode: standard SFT"
fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$REPO_DIR/$f" ]; then
        echo "ERROR: $f not found." >&2
        exit 1
    fi
done
echo "Train: $(wc -l < "$REPO_DIR/$TRAIN_FILE") examples"
echo "Val:   $(wc -l < "$REPO_DIR/$VAL_FILE") examples"

ADAPTER_DIR="$REPO_DIR/data/models/gptoss_lora_$TAG"
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

if [ -n "${VIRTUAL_ENV:-}" ]; then
    NVIDIA_LIB_ROOT="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia"
    if [ -d "$NVIDIA_LIB_ROOT" ]; then
        while IFS= read -r -d '' libdir; do
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
        done < <(find "$NVIDIA_LIB_ROOT" -maxdepth 2 -type d -name lib -print0)
    fi
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== LoRA Fine-tuning: GPT-OSS 120B (MXFP4→bf16 dequant)  (TAG=$TAG) ==="

python echr/finetune/finetune_echr.py \
    --model_id openai/gpt-oss-120b \
    --train_dataset "$REPO_DIR/$TRAIN_FILE" \
    --val_dataset   "$REPO_DIR/$VAL_FILE" \
    --output_dir    "$ADAPTER_DIR" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_len "${MAX_SEQ_LEN:-4096}" \
    --no_eval \
    --save_steps 25 \
    --resume_from_checkpoint \
    ${HARMONY_FLAG}

echo ""
echo "Done: $(date). Adapter → $ADAPTER_DIR/adapter_final"
