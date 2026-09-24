#!/bin/bash
# Resume all non-RR Llama prediction jobs (uses --resume, safe to re-run).
# Run this once the Llama server is up and endpoint file is updated.
#
# Usage:
#   bash scripts/resume_llama_nonrr.sh
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
echo "Endpoint OK. Submitting jobs..."

LLAMA_MODEL="Llama-3.3-70B"

echo "--- Article 3: zero-shot, few-shot, iterative ---"
for cond in base cot; do
    COT_FLAG=$( [ "$cond" = "cot" ] && echo "true" || echo "false" )
    jid=$(sbatch --export=ARTICLE=3,MODEL=$LLAMA_MODEL,CONDITION=base,COT=$COT_FLAG,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art3 ${cond}: job $jid"
done
jid=$(sbatch --export=ARTICLE=3,MODEL=$LLAMA_MODEL,CONDITION=court,COT=false,ENDPOINT=$LLAMA_EP \
    scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
echo "  Art3 court: job $jid"
jid=$(sbatch --export=ARTICLE=3,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
    scripts/predict_few_shot.sh 2>&1 | grep -oP '\d+')
echo "  Art3 few_shot: job $jid"
jid=$(sbatch --export=ARTICLE=3,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
    scripts/predict_iterative.sh 2>&1 | grep -oP '\d+')
echo "  Art3 iterative: job $jid"
sleep 5

echo "--- Article 6: zero-shot, few-shot, iterative ---"
for cond in base cot; do
    COT_FLAG=$( [ "$cond" = "cot" ] && echo "true" || echo "false" )
    jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,CONDITION=base,COT=$COT_FLAG,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art6 ${cond}: job $jid"
done
jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,CONDITION=court,COT=false,ENDPOINT=$LLAMA_EP \
    scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
echo "  Art6 court: job $jid"
jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
    scripts/predict_few_shot.sh 2>&1 | grep -oP '\d+')
echo "  Art6 few_shot: job $jid"
jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
    scripts/predict_iterative.sh 2>&1 | grep -oP '\d+')
echo "  Art6 iterative: job $jid"
sleep 5

echo "--- Article 8: zero-shot, few-shot, iterative ---"
for cond in base cot; do
    COT_FLAG=$( [ "$cond" = "cot" ] && echo "true" || echo "false" )
    jid=$(sbatch --export=ARTICLE=8,MODEL=$LLAMA_MODEL,CONDITION=base,COT=$COT_FLAG,ENDPOINT=$LLAMA_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art8 ${cond}: job $jid"
done
jid=$(sbatch --export=ARTICLE=8,MODEL=$LLAMA_MODEL,CONDITION=court,COT=false,ENDPOINT=$LLAMA_EP \
    scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
echo "  Art8 court: job $jid"
jid=$(sbatch --export=ARTICLE=8,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
    scripts/predict_few_shot.sh 2>&1 | grep -oP '\d+')
echo "  Art8 few_shot: job $jid"
jid=$(sbatch --export=ARTICLE=8,MODEL=$LLAMA_MODEL,ENDPOINT=$LLAMA_EP \
    scripts/predict_iterative.sh 2>&1 | grep -oP '\d+')
echo "  Art8 iterative: job $jid"
sleep 5

echo "--- RAG Art3 (faiss, bm25, tgn_kg, gold) ---"
for ret in faiss bm25 tgn_kg gold; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=3,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art3 ${ret} k${k}: job $jid"
    done
done
sleep 5

echo "--- RAG Art6 (faiss, bm25, tgn_kg, gold) ---"
for ret in faiss bm25 tgn_kg gold; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=6,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art6 ${ret} k${k}: job $jid"
    done
done
sleep 5

echo "--- RAG Art8 (faiss, bm25, tgn_kg, gold) ---"
for ret in faiss bm25 tgn_kg gold; do
    for k in 3 5 10; do
        jid=$(sbatch --export=ARTICLE=8,MODEL=$LLAMA_MODEL,RETRIEVER=$ret,K=$k,ENDPOINT=$LLAMA_EP \
            scripts/predict_retrieval.sh 2>&1 | grep -oP '\d+')
        echo "  Art8 ${ret} k${k}: job $jid"
    done
done

echo ""
echo "=== resume_llama_nonrr complete (51 jobs). ==="
echo "    +RR jobs: run scripts/phase2_llama_resubmit.sh (+RR section) once non-RR jobs are stable."
