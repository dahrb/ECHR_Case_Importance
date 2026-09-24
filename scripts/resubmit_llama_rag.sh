#!/bin/bash
# Resubmit all Llama GOLD/KG/FAISS/BM25 RAG jobs with corrected --endpoint flag.
# Run AFTER cancelling jobs 10419841-10419870 and clearing corrupted result files.
#
# Usage: bash scripts/resubmit_llama_rag.sh

set -euo pipefail

LLAMA_EP="http://gpu41.barkla2.liv.alces.network:8002/v1"
LLAMA_MODEL="Llama-3.3-70B"

echo "=== Step 1: Clear corrupted Llama RAG result files ==="
for art in 3 6 8; do
    for ret in gold tgn_kg faiss bm25; do
        for k in 3 5 10; do
            f="data/results/article${art}/retrieval_${ret}_k${k}_text1_test_${LLAMA_MODEL}.jsonl"
            if [ -f "$f" ]; then
                rm "$f"
                echo "  Removed $f"
            fi
        done
    done
done

echo ""
echo "=== Step 2: Resubmit GOLD k=3/5/10 for Art 3/6/8 ==="
for art in 3 6 8; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=${art},MODEL=${LLAMA_MODEL},RETRIEVER=gold,K=${k},ENDPOINT=${LLAMA_EP} \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art${art} GOLD k${k}: job $jid"
    done
done

echo ""
echo "=== Step 3: Resubmit KG k=3/5/10 for Art 3/6/8 ==="
for art in 3 6 8; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=${art},MODEL=${LLAMA_MODEL},RETRIEVER=tgn_kg,K=${k},ENDPOINT=${LLAMA_EP} \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art${art} KG k${k}: job $jid"
    done
done

echo ""
echo "=== Step 4: Resubmit FAISS k=3/5/10 for Art 3/8 (Art6 blocked on summaries) ==="
for art in 3 8; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=${art},MODEL=${LLAMA_MODEL},RETRIEVER=faiss,K=${k},ENDPOINT=${LLAMA_EP} \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art${art} FAISS k${k}: job $jid"
    done
done

echo ""
echo "=== Step 5: Resubmit BM25 k=3/5/10 for Art 3/8 ==="
for art in 3 8; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=${art},MODEL=${LLAMA_MODEL},RETRIEVER=bm25,K=${k},ENDPOINT=${LLAMA_EP} \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art${art} BM25 k${k}: job $jid"
    done
done

echo ""
echo "Done. All 30 Llama RAG jobs resubmitted with correct endpoint."
