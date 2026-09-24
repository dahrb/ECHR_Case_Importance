#!/bin/bash
# Cancel all running Llama prediction jobs, restart Llama server on H100,
# then resubmit all incomplete Llama jobs with --resume + new endpoint.
# Run this when the current Llama server (10419825) is about to expire or has died.
# Usage: bash scripts/restart_llama_server.sh

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

echo "=== Step 1: Cancel all running prediction jobs ==="
LLAMA_PIDS=$(squeue -u sgdbareh --format="%i %j" --noheader 2>/dev/null \
    | grep -E 'predict_|summarize' | awk '{print $1}' | tr '\n' ' ')
if [ -n "$LLAMA_PIDS" ]; then
    echo "  Cancelling jobs: $LLAMA_PIDS"
    scancel $LLAMA_PIDS
    sleep 5
else
    echo "  No prediction jobs to cancel."
fi

echo ""
echo "=== Step 2: Start new Llama server on H100 ==="
# Script header already has --partition=gpu-h100; no override needed.
# Use .venv (FA2) via VENV_NAME override.
NEW_SRV=$(sbatch --export=MODEL=nvidia/Llama-3.3-70B-Instruct-FP8,MODEL_NAME=Llama-3.3-70B,PORT=8002,TP=2,MAX_MODEL_LEN=30000,VENV_NAME=.venv \
    scripts/start_vllm_server.sh 2>&1 | grep -oP '\d+')
echo "  New Llama server: job $NEW_SRV"

echo ""
echo "=== Step 3: Wait for server to start and write endpoint ==="
echo "  Waiting up to 20 minutes for endpoint file to update..."
ENDPOINT_FILE="$REPO_DIR/data/vllm_Llama-3.3-70B_endpoint.txt"
OLD_EP=$(cat "$ENDPOINT_FILE" 2>/dev/null || echo "none")
for i in $(seq 1 40); do
    sleep 30
    NEW_EP=$(cat "$ENDPOINT_FILE" 2>/dev/null || echo "none")
    if [ "$NEW_EP" != "$OLD_EP" ] && [ "$NEW_EP" != "none" ]; then
        echo "  Server up at: $NEW_EP"
        break
    fi
    echo "  Still waiting... ($((i*30))s)"
done
LLAMA_EP=$(cat "$ENDPOINT_FILE")
echo "  Using endpoint: $LLAMA_EP"

echo ""
echo "=== Step 4: Resubmit Llama prediction jobs with --resume ==="
LLAMA_MODEL="Llama-3.3-70B"
ART3_RR="data/vectordb/article3/rerank_model_2026-09-04_00-24-35_fold_3"
ART8_RR="data/vectordb/article8/rerank_model_2026-09-04_00-24-35_fold_2"

echo "--- Zero-shot (base, cot, court) ---"
for art in 3 6 8; do
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,CONDITION=base,COT=false,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} base Llama: job $jid"
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,CONDITION=base,COT=true,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} cot Llama: job $jid"
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,CONDITION=court,COT=false,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} court Llama: job $jid"
done

echo "--- Few-shot ---"
for art in 3 6 8; do
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
        scripts/predict_few_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} few_shot Llama: job $jid"
done

echo "--- Iterative ---"
for art in 3 6 8; do
    jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
        scripts/predict_iterative.sh 2>&1 | grep -oP '\d+')
    echo "  Art${art} iterative Llama: job $jid"
done

echo "--- RAG (FAISS, BM25, KG, Gold) Art3+Art8 ---"
for art in 3 8; do
    for ret in faiss bm25 tgn_kg gold; do
        for k in 3 5 10; do
            jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
                scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
            echo "  Art${art} ${ret} k${k} Llama: job $jid"
        done
    done
done

echo "--- RAG (KG, Gold) Art6 ---"
for ret in tgn_kg gold; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret} k${k} Llama: job $jid"
    done
done

echo "--- Art3+Art8 Rerank predictions ---"
for art in 3 8; do
    if [ "$art" = "3" ]; then RR_MODEL="$ART3_RR"; else RR_MODEL="$ART8_RR"; fi
    for ret in faiss bm25 tgn_kg; do
        for k in 3 5 10; do
            jid=$(sbatch --export=ARTICLE=$art,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,RERANK=1,RERANK_MODEL=$RR_MODEL,ENDPOINT=$LLAMA_EP \
                scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
            echo "  Art${art} ${ret}+RR k${k} Llama: job $jid"
        done
    done
done

echo ""
echo "=== Done — Llama server restart complete ==="
echo "Cancel old server if still alive: scancel 10419825"
echo "New endpoint: $LLAMA_EP"
