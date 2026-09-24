#!/bin/bash
# Phase 2: read new Llama endpoint and resubmit all prediction jobs.
set -euo pipefail
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

ENDPOINT_FILE="$REPO_DIR/data/vllm_Llama-3.3-70B_endpoint.txt"
LLAMA_EP=$(cat "$ENDPOINT_FILE")
echo "Using endpoint: $LLAMA_EP"

# Verify the endpoint is actually responding before submitting any jobs
echo "Verifying endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$LLAMA_EP/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Endpoint $LLAMA_EP not responding (HTTP $HTTP_CODE). Aborting." >&2
    exit 1
fi
echo "Endpoint OK (HTTP $HTTP_CODE). Submitting jobs..."

LLAMA_MODEL="Llama-3.3-70B-Instruct-FP8"
ART3_RR="data/vectordb/article3/rerank_model_2026-09-15_11-46-42_fold_2"
ART8_RR="data/vectordb/article8/rerank_model_2026-09-15_11-46-42_fold_5"

echo "--- Zero-shot (base, cot, court) ---"
for art in 3 6 8; do
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,CONDITION=base,COT=false,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} base: job $jid"
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,CONDITION=base,COT=true,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} cot: job $jid"
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,CONDITION=court,COT=false,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} court: job $jid"
done

echo "--- Few-shot ---"
for art in 3 6 8; do
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
        scripts/predict_few_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} few_shot: job $jid"
done

echo "--- Iterative ---"
for art in 3 6 8; do
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
        scripts/predict_iterative.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} iterative: job $jid"
done

echo "--- RAG Art3+Art8 (FAISS, BM25, KG, Gold) ---"
for art in 3 8; do
    for ret in faiss bm25 tgn_kg gold; do
        for k in 3 5 10; do
            jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
                scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
            echo "  Art${art} ${ret} k${k}: job $jid"
        done
    done
done

echo "--- RAG Art6 (KG, Gold only — FAISS/BM25 handled separately) ---"
for ret in tgn_kg gold; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret} k${k}: job $jid"
    done
done

echo "--- Art6 FAISS+BM25 (new, first run) ---"
for ret in faiss bm25; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret} k${k}: job $jid"
    done
done

echo "--- Art3+Art8 +RR ---"
for art in 3 8; do
    if [ "$art" = "3" ]; then RR_MODEL="$ART3_RR"; else RR_MODEL="$ART8_RR"; fi
    for ret in faiss bm25 tgn_kg; do
        for k in 3 5 10; do
            jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$RR_MODEL,ENDPOINT=$LLAMA_EP \
                scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
            echo "  Art${art} ${ret}+RR k${k}: job $jid"
        done
    done
done

echo "--- Art6 +RR ---"
ART6_RR="data/vectordb/article6/rerank_model_2026-09-15_11-46-42_fold_1"
for ret in faiss bm25 tgn_kg; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$ART6_RR,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret}+RR k${k}: job $jid"
    done
done

echo ""
echo "=== Phase 2 complete. All Llama prediction jobs resubmitted. ==="
