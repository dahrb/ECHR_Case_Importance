#!/bin/bash -l
# Merge one v2 GPT-OSS-120B LoRA epoch checkpoint into a standalone model.
# Loads the MXFP4 base as bf16 and merges via PEFT.  Set MXFP4_REQUANTIZE=1
# on Hopper to requantize in the same job; REQUIRE_MXFP4=1 makes any fallback
# to BF16 a hard failure.
# Output: data/models_v2/gptoss_merged_art{ARTICLE}_e{EPOCH}/ + FORMAT sentinel.
#
# Usage:
#   sbatch --export=ARTICLE=3,EPOCH=1 scripts/merge_gptoss_ft.sh
#
#SBATCH --job-name=gptoss_merge
#SBATCH --output=data/data_collection/logs/gptoss_merge_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_merge_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --no-requeue

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"
: "${EPOCH:?Must set EPOCH (1 or 2) via --export=EPOCH=N}"
case "$ARTICLE:$EPOCH" in
    3:1) STEP=21 ;; 3:2) STEP=42 ;;
    6:1) STEP=62 ;; 6:2) STEP=124 ;;
    8:1) STEP=34 ;; 8:2) STEP=68 ;;
    *) echo "ERROR: ARTICLE must be 3/6/8 and EPOCH must be 1/2" >&2; exit 2 ;;
esac

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/.venv_11_2"
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
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" XDG_CACHE_HOME="$HF_HOME/xdg_cache"
# ModelOpt's MXFP4 export needs a transient allocation after merging.  Keep
# allocator segments expandable so cache fragmentation does not turn otherwise
# available A100 memory into an export OOM.
export VLLM_MXFP4_DEQUANT=1 PYTORCH_ALLOC_CONF=expandable_segments:True
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Explicit overrides allow the current validated adapter_final artifacts to be
# merged without pretending they are an older epoch checkpoint.
ADAPTER_DIR="${ADAPTER_DIR:-$REPO_DIR/data/models_v2/gptoss_lora_art${ARTICLE}_2ep_noeval/checkpoint-${STEP}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/models_v2/gptoss_merged_art${ARTICLE}_e${EPOCH}}"
CALIB_DATA="$REPO_DIR/data/finetune_v2/art${ARTICLE}/sft_train.jsonl"

echo "==== Merge GPT-OSS v2 art${ARTICLE} epoch ${EPOCH} (checkpoint-${STEP}) ===="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MERGE_ARGS=(
    --article "$ARTICLE"
    --adapter_dir "$ADAPTER_DIR"
    --output_dir "$OUTPUT_DIR"
    --calib_data "$CALIB_DATA"
)
if [ "${MXFP4_REQUANTIZE:-0}" != "1" ]; then
    MERGE_ARGS+=(--skip_mxfp4)
fi

python echr/finetune/merge_lora_gptoss.py \
    "${MERGE_ARGS[@]}"

echo "==== Merge complete: art${ARTICLE} epoch ${EPOCH} ===="; date
FORMAT=$(cat "$OUTPUT_DIR/FORMAT" 2>/dev/null || echo "unknown")
echo "Saved format: $FORMAT"
if [ "${REQUIRE_MXFP4:-0}" = "1" ] && [ "$FORMAT" != "mxfp4" ]; then
    echo "ERROR: required MXFP4 output, got $FORMAT" >&2
    exit 1
fi
