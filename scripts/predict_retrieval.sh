#!/bin/bash
# Run retrieval-augmented (RAG) importance predictions using BM25 or FAISS top-K.
#
# Usage:
#   sbatch --export=ARTICLE=3 scripts/predict_retrieval.sh
#   sbatch --export=ARTICLE=3,RETRIEVER=faiss,K=5 scripts/predict_retrieval.sh
#
# Optional overrides:
#   RETRIEVER: bm25 (default) or faiss or tgn_kg
#   K: number of retrieved examples (default 3)
#   MODEL: vLLM model name (default gpt-oss-120b)
#   TEXT: 1=Subject Matter (default), 2=Questions, 3=Both
#   SPLIT: test (default) or valid
#   RERANK: 1 to enable LegalBERT cross-encoder reranking
#   RERANK_MODEL: path to saved CrossEncoder model dir (required if RERANK=1)
#   RERANK_POOL: candidate pool size before reranking (default 50)
#   PROMPT_DIR: complete pre-materialized prompt shard directory (preferred)
#
#SBATCH --job-name=predict_rag
#SBATCH --output=data/data_collection/logs/predict_rag_art${ARTICLE}_%j.out
#SBATCH --error=data/data_collection/logs/predict_rag_art${ARTICLE}_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=nodes

set -euo pipefail

: "${ARTICLE:?Must set ARTICLE (3, 6, or 8) via --export=ARTICLE=N}"

MODEL="${MODEL:-gpt-oss-120b}"
RETRIEVER="${RETRIEVER:-bm25}"
K="${K:-${SLURM_ARRAY_TASK_ID:-3}}"
TEXT="${TEXT:-1}"
SPLIT="${SPLIT:-test}"
RERANK="${RERANK:-0}"
RERANK_MODEL="${RERANK_MODEL:-}"
RERANK_POOL="${RERANK_POOL:-50}"
MAX_TOKENS="${MAX_TOKENS:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
NO_THINKING="${NO_THINKING:-false}"
ENDPOINT="${ENDPOINT:-}"
ENDPOINT_FILE_PATH="${ENDPOINT_FILE_PATH:-}"
PROMPT_DIR="${PROMPT_DIR:-}"
RUN_TAG="${RUN_TAG:-}"

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
echo "  RAG Predictions — Article $ARTICLE"
echo "  model=$MODEL  retriever=$RETRIEVER  k=$K  text=$TEXT  split=$SPLIT  rerank=$RERANK"
echo "  Started: $(date)"
echo "========================================"

args=(
    --article "$ARTICLE"
    --model "$MODEL"
    --retriever "$RETRIEVER"
    --k "$K"
    --text "$TEXT"
    --split "$SPLIT"
    --max_tokens "$MAX_TOKENS"
    --max_model_len "$MAX_MODEL_LEN"
    --resume
)

if [ "$RERANK" = "1" ]; then
    : "${RERANK_MODEL:?RERANK=1 requires RERANK_MODEL to be set}"
    args+=(--rerank --rerank_model "$RERANK_MODEL" --rerank_pool "$RERANK_POOL")
fi
if [ -n "$REASONING_EFFORT" ]; then
    args+=(--reasoning_effort "$REASONING_EFFORT")
fi
if [ "$NO_THINKING" = "true" ]; then
    args+=(--no_thinking)
fi
if [ -n "$PROMPT_DIR" ]; then
    args+=(--prompt_dir "$PROMPT_DIR")
fi
if [ -n "$RUN_TAG" ]; then
    args+=(--run_tag "$RUN_TAG")
fi
if [ -n "$ENDPOINT" ]; then
    args+=(--endpoint "$ENDPOINT")
fi

"$VENV/bin/python" echr/prediction/run_retrieval_predictions.py "${args[@]}"

echo "========================================"
echo "  Done: $(date)"
echo "========================================"
