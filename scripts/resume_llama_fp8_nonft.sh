#!/bin/bash
# Full NON-FT Llama sweep under the FP8 name (Llama-3.3-70B-Instruct-FP8).
# 1) Verifies the FP8 server endpoint is live.
# 2) Cleans null-prediction rows (resume gotcha: null rows are otherwise skipped).
# 3) Submits ALL non-FT conditions with --resume, batched per article.
#
# Prereq: FP8 server up (MODEL_NAME=Llama-3.3-70B-Instruct-FP8) -> endpoint file written.
# Usage: bash scripts/resume_llama_fp8_nonft.sh
set -euo pipefail
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

MODEL="Llama-3.3-70B-Instruct-FP8"
ENDPOINT_FILE="$REPO_DIR/data/vllm_${MODEL}_endpoint.txt"
EP=$(cat "$ENDPOINT_FILE")
echo "Endpoint: $EP"
HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$EP/models" 2>/dev/null || echo 000)
[ "$HTTP" = "200" ] || { echo "ERROR: endpoint not responding (HTTP $HTTP)"; exit 1; }
echo "Endpoint OK."

echo "=== Cleaning null-prediction rows from FP8 non-FT files ==="
python scripts/clean_null_predictions.py --glob "data/results/article*/*_${MODEL}.jsonl"

RR3="data/vectordb/article3/rerank_model_2026-09-15_11-46-42_fold_2"
RR6="data/vectordb/article6/rerank_model_2026-09-15_11-46-42_fold_1"
RR8="data/vectordb/article8/rerank_model_2026-09-15_11-46-42_fold_5"

submit_article () {
    local art=$1 rr=$2
    echo "--- Article $art: zero-shot / few-shot / iterative ---"
    sbatch --export=ARTICLE=$art,MODEL=$MODEL,CONDITION=base,COT=false,ENDPOINT=$EP scripts/predict_zero_shot.sh
    sbatch --export=ARTICLE=$art,MODEL=$MODEL,CONDITION=base,COT=true,ENDPOINT=$EP  scripts/predict_zero_shot.sh
    sbatch --export=ARTICLE=$art,MODEL=$MODEL,CONDITION=court,COT=false,ENDPOINT=$EP scripts/predict_zero_shot.sh
    sbatch --export=ARTICLE=$art,MODEL=$MODEL,ENDPOINT=$EP scripts/predict_few_shot.sh
    sbatch --export=ARTICLE=$art,MODEL=$MODEL,ENDPOINT=$EP scripts/predict_iterative.sh
    echo "--- Article $art: retrieval (faiss/bm25/tgn_kg/gold) k=3/5/10 ---"
    for ret in faiss bm25 tgn_kg gold; do
        for k in 3 5 10; do
            sbatch --export=ARTICLE=$art,MODEL=$MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$EP scripts/predict_retrieval.sh
        done
    done
    echo "--- Article $art: +RR (faiss/bm25/tgn_kg) k=3/5/10 ---"
    for ret in faiss bm25 tgn_kg; do
        for k in 3 5 10; do
            sbatch --export=ARTICLE=$art,MODEL=$MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$rr,ENDPOINT=$EP scripts/predict_retrieval.sh
        done
    done
}

ART="${ART:-all}"
case "$ART" in
    3) submit_article 3 "$RR3" ;;
    6) submit_article 6 "$RR6" ;;
    8) submit_article 8 "$RR8" ;;
    all)
        submit_article 3 "$RR3"; sleep 5
        submit_article 6 "$RR6"; sleep 5
        submit_article 8 "$RR8"
        ;;
    *) echo "ART must be 3|6|8|all"; exit 1 ;;
esac
echo "=== FP8 non-FT submission complete (ART=$ART). 26 conditions/article. ==="
