#!/bin/bash
# Submit ONLY the +RR Llama jobs (Art3/6/8, faiss/bm25/tgn_kg, k=3/5/10).
# Run AFTER resume_llama_nonrr.sh has been launched and non-RR jobs are stable.
#
# Usage:
#   bash scripts/run_llama_rr_only.sh
set -euo pipefail
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

ENDPOINT_FILE="$REPO_DIR/data/vllm_Llama-3.3-70B_endpoint.txt"
LLAMA_EP=$(cat "$ENDPOINT_FILE")
echo "Using endpoint: $LLAMA_EP"

echo "Verifying endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$LLAMA_EP/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Endpoint $LLAMA_EP not responding (HTTP $HTTP_CODE). Aborting." >&2
    exit 1
fi
echo "Endpoint OK. Submitting +RR jobs..."

LLAMA_MODEL="Llama-3.3-70B"
ART3_RR="data/vectordb/article3/rerank_model_2026-09-15_11-46-42_fold_2"
ART6_RR="data/vectordb/article6/rerank_model_2026-09-15_11-46-42_fold_1"
ART8_RR="data/vectordb/article8/rerank_model_2026-09-15_11-46-42_fold_5"

echo "--- Art3 +RR ---"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=3,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART3_RR,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art3 ${ret}+RR k${k}: job $jid"
    done
done
sleep 5

echo "--- Art6 +RR ---"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART6_RR,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret}+RR k${k}: job $jid"
    done
done
sleep 5

echo "--- Art8 +RR ---"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=8,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART8_RR,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art8 ${ret}+RR k${k}: job $jid"
    done
done

echo ""
echo "=== run_llama_rr_only complete (27 +RR jobs submitted). ==="
