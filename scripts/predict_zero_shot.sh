#!/bin/bash
# Run zero-shot predictions on comm_test cases using GPT-OSS via vLLM.
# Requires the vLLM server to be running (start with scripts/start_vllm_server.sh).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/predict_zero_shot.sh
#   sbatch --export=ARTICLE=6 scripts/predict_zero_shot.sh
#   sbatch --export=ARTICLE=8 scripts/predict_zero_shot.sh
#
# Optional overrides:
#   --export=ARTICLE=3,MODEL=gpt-oss-20b,TEXT=1,CONDITION=base
#   TEXT: 1=Subject Matter (default), 2=Questions, 3=Both
#   CONDITION: base (default, importance 1-4) | court (Judgment/Decision)
#   COT: true | false (default)
#
#SBATCH --job-name=predict_base
#SBATCH --output=data/data_collection/logs/predict_base_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/predict_base_art${ARTICLE}_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

MODEL="${MODEL:-gpt-oss-20b}"
TEXT="${TEXT:-1}"
SPLIT="${SPLIT:-test}"
COT="${COT:-false}"
CONDITION="${CONDITION:-base}"
MAX_TOKENS="${MAX_TOKENS:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
NO_THINKING="${NO_THINKING:-false}"
ENDPOINT="${ENDPOINT:-}"
ENDPOINT_FILE_PATH="${ENDPOINT_FILE_PATH:-}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs data/results/article"$ARTICLE"

# If no explicit endpoint, wait for endpoint file to be written by server job
if [ -z "$ENDPOINT" ] && [ -n "$ENDPOINT_FILE_PATH" ]; then
    echo "Waiting for endpoint file: $ENDPOINT_FILE_PATH"
    for i in $(seq 1 360); do
        [ -f "$ENDPOINT_FILE_PATH" ] && ENDPOINT=$(cat "$ENDPOINT_FILE_PATH") && break
        echo "  File not ready yet (${i}/360), sleeping 10s..."
        sleep 10
    done
fi

# Wait for vLLM server to be healthy before starting predictions
if [ -n "$ENDPOINT" ]; then
    BASE_URL="${ENDPOINT%/v1}"
    echo "Waiting for server at $BASE_URL/health ..."
    for i in $(seq 1 90); do
        curl -sf "$BASE_URL/health" > /dev/null 2>&1 && { echo "  Server healthy (checked after ${i}x10s)"; break; }
        [ $i -eq 90 ] && { echo "  WARNING: server not healthy after 900s, proceeding anyway"; break; }
        sleep 10
    done
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Zero-Shot Predictions — Article $ARTICLE"
echo "  model=$MODEL  text=$TEXT  condition=$CONDITION"
echo "  Started: $(date)"
echo "========================================"

"$VENV/bin/python" echr/prediction/run_predictions.py \
    --article "$ARTICLE" \
    --model "$MODEL" \
    --text "$TEXT" \
    --condition "$CONDITION" \
    --split "$SPLIT" \
    --max_tokens "$MAX_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    ${REASONING_EFFORT:+--reasoning_effort "$REASONING_EFFORT"} \
    $([ "$COT" = "true" ] && echo "--cot") \
    $([ "$NO_THINKING" = "true" ] && echo "--no_thinking") \
    ${ENDPOINT:+--endpoint "$ENDPOINT"} \
    --resume

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
