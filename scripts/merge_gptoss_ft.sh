#!/bin/bash -l
# Merge GPT-OSS-120B LoRA adapter for one article into a standalone model.
# Loads MXFP4 base as bf16, merges via PEFT, attempts MXFP4 requantization.
# Output: data/models/gptoss_merged_art{ARTICLE}/  + FORMAT sentinel file.
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/merge_gptoss_ft.sh
#
#SBATCH --job-name=gptoss_merge
#SBATCH --output=data/data_collection/logs/gptoss_merge_art%a_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_merge_art%a_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

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
export VLLM_MXFP4_DEQUANT=1
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

ADAPTER_DIR="$REPO_DIR/data/models/gptoss_lora_art${ARTICLE}/adapter_final"
OUTPUT_DIR="$REPO_DIR/data/models/gptoss_merged_art${ARTICLE}"

echo "==== Merge GPT-OSS art${ARTICLE} ===="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python echr/finetune/merge_lora_gptoss.py \
    --article "$ARTICLE" \
    --adapter_dir "$ADAPTER_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "==== Merge complete: art${ARTICLE} ===="; date
FORMAT=$(cat "$OUTPUT_DIR/FORMAT" 2>/dev/null || echo "unknown")
echo "Saved format: $FORMAT"
