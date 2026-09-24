#!/bin/bash -l
# Generate GPT-OSS reasoning traces for harmony SFT retraining (all 3 articles).
# Polls data/vllm_gptoss_base_endpoint.txt until the base server is up, then
# calls gen_reasoning_traces.py which generates traces_train.jsonl + traces_val.jsonl
# for art3, art6, art8.
#
# Submit AFTER (or together with) serve_gptoss_base.sh — the Python script polls.
#
#SBATCH --job-name=gen_gptoss_traces
#SBATCH --output=data/data_collection/logs/gen_traces_%j.out
#SBATCH --error=data/data_collection/logs/gen_traces_%j.err
#SBATCH --partition=lowpriority
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
ADM_DIR="/users/sgdbareh/scratch/ADM_JURIX"
VENV="$ADM_DIR/.venv_11_2"

cd "$REPO_DIR"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== GPT-OSS Reasoning Trace Generation ===" ; date
hostname

python echr/finetune/gen_reasoning_traces.py \
    --articles art3 art6 art8 \
    --data_dir data/finetune \
    --endpoint_file data/vllm_gptoss_base_endpoint.txt \
    --model gpt-oss-120b \
    --max_workers 8 \
    --max_tokens 2000

echo "" ; echo "=== Done: $(date) ==="
