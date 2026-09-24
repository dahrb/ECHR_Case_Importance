#!/bin/bash
# Downloads Qwen/Qwen3-Embedding-8B weights to the project model cache.
# Run once from the login node (no GPU needed) before building FAISS indexes.
#
#   sbatch scripts/download_qwen_embedding.sh
#
#SBATCH --job-name=download_qwen_emb
#SBATCH --output=data/data_collection/logs/download_qwen_emb_%j.out
#SBATCH --error=data/data_collection/logs/download_qwen_emb_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=lowpriority

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
MODEL_CACHE="$REPO_DIR/models/hub"

mkdir -p "$MODEL_CACHE"
mkdir -p data/data_collection/logs

export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"

echo "Downloading Qwen/Qwen3-Embedding-8B -> $MODEL_CACHE"
echo "Started: $(date)"

"$VENV/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-8B",
    cache_dir="/users/sgdbareh/scratch/ECHR_Importance/models/hub",
)
print("Download complete.")
EOF

echo "Done: $(date)"
