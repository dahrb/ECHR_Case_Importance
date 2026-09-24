#!/bin/bash
# Submit Art6 rerank predictions for GPT-OSS and Llama once Art6 reranker is trained.
# Usage: bash scripts/launch_art6_rerank_preds.sh <ART6_RR_MODEL_PATH> [LLAMA_EP]
# e.g.:  bash scripts/launch_art6_rerank_preds.sh data/vectordb/article6/rerank_model_XXXX_fold_N
set -euo pipefail
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

ART6_RR="${1:?Usage: $0 <rerank_model_path> [llama_endpoint]}"
LLAMA_EP="${2:-$(cat data/vllm_Llama-3.3-70B_endpoint.txt 2>/dev/null)}"
LLAMA_MODEL="Llama-3.3-70B"

echo "Art6 RR model: $ART6_RR"
echo "Llama endpoint: $LLAMA_EP"
echo ""

echo "=== Art6 GPT-OSS +RR predictions ==="
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART6_RR \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret}+RR k${k} GPT-OSS: job $jid"
    done
done

echo ""
echo "=== Art6 Llama +RR predictions ==="
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART6_RR,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret}+RR k${k} Llama: job $jid"
    done
done
echo ""
echo "=== Art6 +RR predictions submitted (18 jobs total) ==="
