#!/bin/bash
# Submit GPT-OSS train-split inference jobs: base + iterative for each article.
# Must be run AFTER verifying GPT-OSS endpoint is healthy.
# Art3 submitted first (priority for fine-tuning), then Art6/Art8.
#
# Usage:
#   bash scripts/run_gptoss_train_inference.sh
#   bash scripts/run_gptoss_train_inference.sh 3        # Art3 only
#   bash scripts/run_gptoss_train_inference.sh 3,6,8   # all (default)
set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

ENDPOINT_FILE="$REPO_DIR/data/vllm_gptoss_endpoint.txt"
GPTOSS_EP=$(cat "$ENDPOINT_FILE")
MODEL="gpt-oss-120b"

echo "GPT-OSS endpoint: $GPTOSS_EP"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$GPTOSS_EP/models" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: GPT-OSS endpoint not responding (HTTP $HTTP_CODE). Aborting." >&2
    exit 1
fi
echo "Endpoint OK."

ARTICLES="${1:-3,6,8}"
IFS=',' read -ra ART_LIST <<< "$ARTICLES"

for ARTICLE in "${ART_LIST[@]}"; do
    echo ""
    echo "--- Article $ARTICLE: base (zero-shot train) ---"
    jid=$(sbatch --export=ARTICLE=$ARTICLE,MODEL=$MODEL,SPLIT=train,CONDITION=base,COT=false,ENDPOINT=$GPTOSS_EP \
        scripts/predict_zero_shot.sh 2>&1 | grep -oP '\d+')
    echo "  Art${ARTICLE} base train: job $jid"

    echo "--- Article $ARTICLE: iterative (train) ---"
    jid=$(sbatch --export=ARTICLE=$ARTICLE,MODEL=$MODEL,SPLIT=train,ENDPOINT=$GPTOSS_EP \
        scripts/predict_iterative.sh 2>&1 | grep -oP '\d+')
    echo "  Art${ARTICLE} iterative train: job $jid"

    sleep 3
done

echo ""
echo "All train-inference jobs submitted."
echo "Once complete, run: python echr/prediction/create_finetune_data.py --articles $ARTICLES"
