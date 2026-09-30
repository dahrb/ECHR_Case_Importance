#!/bin/bash
# Hourly overnight status monitor. It audits queue health, endpoint health,
# output completeness, and null/error growth; it cannot wake Codex.
#SBATCH --job-name=monitor_kg
#SBATCH --output=data/data_collection/logs/monitor_kg_%j.out
#SBATCH --error=data/data_collection/logs/monitor_kg_%j.err
#SBATCH --partition=nodes
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

REPO_DIR="/users/sgdbareh/scratch/ECHR_Importance"
VENV="/mnt/data1/users/sgdbareh/venvs/ECHR_Importance"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

for tick in $(seq 1 48); do
    echo "===== KG monitor: $(date --iso-8601=seconds), tick $tick/48 ====="
    squeue -u "$USER" -o '%.18i %.18j %.10T %.12M %.28R' || true
    "$VENV/bin/python" - <<'PY'
import os
import json
import pandas as pd
for article in (3, 6, 8):
    expected = len(pd.read_pickle(f"data/processed/article{article}/splits/comm_test.pkl"))
    counts = []
    for k in (3, 5, 10):
        path = f"data/vectordb/article{article}/tgn_kg_k{k}_results.pkl"
        if not os.path.exists(path):
            counts.append(f"k{k}=missing")
            continue
        data = pd.read_pickle(path)
        counts.append(f"k{k}={len(data)}/{expected},nonempty={sum(bool(v) for v in data.values())}")
    print(f"article{article}: " + "; ".join(counts))
PY
    for endpoint in data/vllm_llama_kg_endpoint.txt data/vllm_llama_kg_endpoint_a100.txt data/vllm_llama_kg_endpoint_bf16_a100.txt data/vllm_llama_kg_endpoint_bf16_a100_2.txt data/vllm_gptoss_base_endpoint.txt; do
        if [ -f "$endpoint" ]; then
            url=$(cat "$endpoint")
            printf 'endpoint %s: ' "$url"
            curl -sf "${url%/v1}/health" >/dev/null && echo healthy || echo unavailable
        fi
    done
    active=$(squeue -h -u "$USER" -n predict_rag -t R -o '%i' | wc -l | tr -d ' ')
    echo "active predict_rag jobs: $active (capacity target: <=6; three per Llama server)"
    if [ "$active" -gt 6 ]; then
        echo "WARNING: prediction concurrency exceeds the two-server operating cap"
    fi
    "$VENV/bin/python" - <<'PY'
import glob, json, os
for p in sorted(glob.glob("data/results/article[368]/retrieval_tgn_kg_k*_text1_test_Llama-3.3-70B-Instruct.jsonl")):
    valid = nulls = 0
    try:
        for line in open(p):
            try:
                v = json.loads(line).get("prediction")
                if v is not None and str(v).strip() in {"1", "2", "3", "4"}: valid += 1
                else: nulls += 1
            except Exception:
                nulls += 1
        print(f"predictions {p}: valid={valid} null_or_error={nulls}")
    except OSError as e:
        print(f"predictions {p}: unreadable={e}")
PY
    [ "$tick" -lt 48 ] && sleep 3600
done
