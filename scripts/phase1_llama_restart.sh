#!/bin/bash
# Phase 1 of Llama server restart: cancel all prediction jobs and submit new H100 server.
# Run at ~03:07 BST Sep 5 when old server has ~40 min left.
# Phase 2 (resubmit prediction jobs) happens at 04:07 poll once server is up.
set -euo pipefail
REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
cd "$REPO_DIR"

echo "=== Phase 1: Cancel all prediction jobs ==="
PIDS=$(squeue -u sgdbareh --format="%i %j" --noheader 2>/dev/null \
    | grep -E 'predict_' | awk '{print $1}' | tr '\n' ' ')
if [ -n "$PIDS" ]; then
    echo "  Cancelling: $PIDS"
    scancel $PIDS
else
    echo "  No prediction jobs running."
fi

echo ""
echo "=== Phase 2: Cancel old Llama server ==="
scancel 10419825 2>/dev/null && echo "  Cancelled 10419825" || echo "  Already gone"

echo ""
echo "=== Phase 3: Submit new Llama server on H100 ==="
NEW_SRV=$(sbatch \
    --export=MODEL=nvidia/Llama-3.3-70B-Instruct-FP8,MODEL_NAME=Llama-3.3-70B,PORT=8002,TP=2,MAX_MODEL_LEN=30000,VENV_NAME=.venv \
    scripts/start_vllm_server.sh 2>&1 | grep -oP '\d+')
echo "  New Llama server job: $NEW_SRV"
echo ""
echo "=== Phase 1 complete. Wait for server to start, then run phase2_llama_resubmit.sh ==="
