#!/bin/bash
# Submit all GPT-OSS +RR prediction jobs (Art3/6/8, all retrievers, all k).
# Run after GPT-OSS server is up and endpoint file is updated.
#
# Usage:
#   bash scripts/run_gptoss_rerank.sh
set -euo pipefail
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

ENDPOINT_FILE="$REPO_DIR/data/vllm_gptoss_endpoint.txt"
GPTOSS_EP=$(cat "$ENDPOINT_FILE")
echo "Using endpoint: $GPTOSS_EP"

echo "Verifying endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$GPTOSS_EP/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Endpoint $GPTOSS_EP not responding (HTTP $HTTP_CODE). Aborting." >&2
    exit 1
fi
echo "Endpoint OK. Submitting +RR jobs..."

ART3_RR="data/vectordb/article3/rerank_model_2026-09-15_11-46-42_fold_2"
ART6_RR="data/vectordb/article6/rerank_model_2026-09-15_11-46-42_fold_1"
ART8_RR="data/vectordb/article8/rerank_model_2026-09-15_11-46-42_fold_5"

echo "--- Art3 GPT-OSS +RR ---"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=3,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART3_RR,ENDPOINT=$GPTOSS_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art3 ${ret}+RR k${k}: job $jid"
    done
done
sleep 5

echo "--- Art6 GPT-OSS +RR ---"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART6_RR,ENDPOINT=$GPTOSS_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret}+RR k${k}: job $jid"
    done
done
sleep 5

echo "--- Art8 GPT-OSS +RR ---"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=8,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART8_RR,ENDPOINT=$GPTOSS_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art8 ${ret}+RR k${k}: job $jid"
    done
done

echo ""
echo "=== run_gptoss_rerank complete (27 jobs submitted). ==="
