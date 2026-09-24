#!/bin/bash
# Launch Art6 FAISS+BM25 retrieval predictions for GPT-OSS only.
# Llama Art6 FAISS/BM25 are submitted separately after Llama server restarts on H100.
# Usage: bash scripts/launch_art6_faiss_bm25.sh

set -euo pipefail

echo "=== Art6 FAISS + BM25 — GPT-OSS ==="
for ret in faiss bm25; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,RETRIEVER=$ret,K=$k scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 $ret k$k GPT-OSS: job $jid"
    done
done

echo ""
echo "=== Art6 Rerank Data Prep ==="
jid=$(sbatch --export=ARTICLE=6 scripts/prepare_rerank_data.sh 2>&1 | grep -oP '\d+')
echo "  prepare_rerank_data: job $jid"
echo "  -> After this completes, run: sbatch --export=ARTICLE=6 scripts/train_rerank.sh"

echo ""
echo "Done. Art6 GPT-OSS FAISS/BM25 predictions + rerank prep submitted."
echo "NOTE: Llama Art6 FAISS/BM25 will be submitted after Llama server restarts on H100."
