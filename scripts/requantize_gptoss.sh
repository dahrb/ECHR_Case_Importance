#!/bin/bash -l
# Re-quantize a merged GPT-OSS bf16 model to MXFP4 using nvidia-modelopt.
# Loads the 218GB bf16 merged model (device_map=auto, 4×A100 80GB),
# runs MXFP4 calibration, exports with export_hf_checkpoint.
# Skips if FORMAT sentinel already says "mxfp4".
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/requantize_gptoss.sh
#
#SBATCH --job-name=gptoss_requant
#SBATCH --output=data/data_collection/logs/gptoss_requant_art%a_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_requant_art%a_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:4

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

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
export HF_HOME HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers" XDG_CACHE_HOME="$HF_HOME/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_ALLOC_CONF=expandable_segments:True
HF_KEY="$ADM_DIR/LLM_Experiments/hf.key"
[ -f "$HF_KEY" ] && { export HF_TOKEN="$(< "$HF_KEY")"; export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"; }
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "==== Requantize GPT-OSS merged art${ARTICLE} bf16 → MXFP4 ===="; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python echr/finetune/requantize_gptoss.py --article "$ARTICLE"

echo "==== Done: $(date) ===="
