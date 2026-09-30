#!/bin/bash
# Run iterative-prompting predictions (Exp 2, §5.3).
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/predict_iterative.sh
#
# Optional overrides:
#   MODEL: vLLM model name (default gpt-oss-120b)
#   TEXT:  1=Subject Matter (default), 2=Questions, 3=Both
#   SPLIT: test (default) or valid
#
#SBATCH --job-name=predict_iter
#SBATCH --output=data/data_collection/logs/predict_iter_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/predict_iter_art${ARTICLE}_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

MODEL="${MODEL:-gpt-oss-120b}"
TEXT="${TEXT:-1}"
SPLIT="${SPLIT:-test}"
ENDPOINT="${ENDPOINT:-}"
ENDPOINT_FILE_PATH="${ENDPOINT_FILE_PATH:-}"
PROMPT_DIR="${PROMPT_DIR:-}"
RUN_TAG="${RUN_TAG:-}"
MAX_TOKENS="${MAX_TOKENS:-600}"
# Iterative evaluation uses medium reasoning; callers may explicitly override it
# for a deliberately labelled ablation only.
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
LEVEL_CONCURRENCY="${LEVEL_CONCURRENCY:-4}"

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"

cd "$REPO_DIR"
mkdir -p data/data_collection/logs data/results/article"$ARTICLE"

# Wait for endpoint file if no direct endpoint given
if [ -z "$ENDPOINT" ] && [ -n "$ENDPOINT_FILE_PATH" ]; then
    echo "Waiting for endpoint file: $ENDPOINT_FILE_PATH"
    for i in $(seq 1 360); do
        [ -f "$ENDPOINT_FILE_PATH" ] && ENDPOINT=$(cat "$ENDPOINT_FILE_PATH") && break
        echo "  Not ready yet (${i}/360), sleeping 10s..."
        sleep 10
    done
fi

# Wait for vLLM server health
if [ -n "$ENDPOINT" ]; then
    BASE_URL="${ENDPOINT%/v1}"
    echo "Waiting for server at $BASE_URL/health ..."
    for i in $(seq 1 90); do
        curl -sf "$BASE_URL/health" > /dev/null 2>&1 && { echo "  Server healthy (${i}x10s)"; break; }
        [ $i -eq 90 ] && echo "  WARNING: server not healthy after 900s, proceeding anyway"
        sleep 10
    done
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "  Iterative-Prompting — Article $ARTICLE"
echo "  model=$MODEL  text=$TEXT  split=$SPLIT"
echo "  prompt_dir=${PROMPT_DIR:-dynamic} level_concurrency=$LEVEL_CONCURRENCY"
echo "  Started: $(date)"
echo "========================================"

ARGS=(
    --article "$ARTICLE" \
    --model "$MODEL" \
    --text "$TEXT" \
    --split "$SPLIT" \
    --resume \
    --max-tokens "$MAX_TOKENS" \
    --reasoning-effort "$REASONING_EFFORT" \
    --level-concurrency "$LEVEL_CONCURRENCY"
)
[ -n "$ENDPOINT" ] && ARGS+=(--endpoint "$ENDPOINT")
[ -n "$PROMPT_DIR" ] && ARGS+=(--prompt-dir "$PROMPT_DIR")
[ -n "$RUN_TAG" ] && ARGS+=(--run-tag "$RUN_TAG")

"$VENV/bin/python" echr/prediction/run_iterative_predictions.py "${ARGS[@]}"

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
