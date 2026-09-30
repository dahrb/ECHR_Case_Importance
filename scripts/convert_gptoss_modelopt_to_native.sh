#!/bin/bash -l
# Convert an existing ModelOpt MXFP4 merged GPT-OSS checkpoint to the native
# GPT-OSS MXFP4 storage layout required by vLLM.
#
# Required environment: ARTICLE=3|6|8

#SBATCH --job-name=gptoss_native_convert
#SBATCH --output=data/data_collection/logs/gptoss_native_convert_%j.out
#SBATCH --error=data/data_collection/logs/gptoss_native_convert_%j.err
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

set -euo pipefail
: "${ARTICLE:?Set ARTICLE=3, 6, or 8}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/.venv_11_2"
BASE_CONFIG="$ADM_DIR/LLM_Models/models/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json"
SOURCE_DIR="${SOURCE_DIR:-$REPO_DIR/data/models_v2/gptoss_merged_art${ARTICLE}_current}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/models_v2/gptoss_merged_art${ARTICLE}_native}"

cd "$REPO_DIR"
module purge; module load cuda/12.8.0-gcc14.2.0 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python echr/finetune/convert_modelopt_gptoss_to_native.py \
  --source "$SOURCE_DIR" --output "$OUTPUT_DIR" --base-config "$BASE_CONFIG"
