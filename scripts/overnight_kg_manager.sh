#!/bin/bash
set -euo pipefail
R=/users/sgdbareh/scratch/ECHR_Importance; V=/mnt/data1/users/sgdbareh/venvs/ECHR_Importance
S="$R/data/data_collection/logs/overnight_kg_manager.state"
cd "$R"; export PYTHONPATH="$R${PYTHONPATH:+:$PYTHONPATH}"
exec >> data/data_collection/logs/overnight_kg_manager.log 2>&1
ep_ok(){ for f in data/vllm_llama_kg_endpoint_bf16_a100_2.txt data/vllm_llama_kg_endpoint_bf16_a100.txt; do [ -f "$f" ] || continue; u=$(cat "$f"); curl -sf --max-time 8 "${u%/v1}/health" >/dev/null 2>&1 && { printf '%s' "$u"; return; }; done; return 1; }
complete(){ "$V/bin/python" - "$1" "$2" <<'PY'
import json,os,sys,pandas as pd
k,rr=sys.argv[1],sys.argv[2]=='1'; ok=True
for a in (3,6,8):
 e=len(pd.read_pickle(f'data/processed/article{a}/splits/comm_test.pkl')); s=f"_k{k}{'_rerank' if rr else ''}_text1_test_Llama-3.3-70B-Instruct.jsonl"; p=f'data/results/article{a}/retrieval_tgn_kg{s}'; n=0
 if not os.path.exists(p): ok=False; continue
 for l in open(p):
  try:
   row=json.loads(l); v=row.get('prediction'); valid=v is not None and str(v).strip() in {'1','2','3','4'}
   permanent=(int(a)==8 and int(k)==5 and row.get('Filename')=='001-115064')
   n+=1 if valid else 0; ok &= valid or permanent
  except: ok=False
 ok &= n==e
print('yes' if ok else 'no')
PY
}
submit(){ k=$1; rr=$2; u=$3; for a in 3 6 8; do f="data/results/article${a}/retrieval_tgn_kg_k${k}$([ "$rr" = 1 ] && echo _rerank)_text1_test_Llama-3.3-70B-Instruct.jsonl"; "$V/bin/python" scripts/clean_null_predictions.py "$f" || true; x=""; [ "$rr" = 1 ] && x=",RERANK=1,RERANK_MODEL=data/vectordb/article${a}/rerank_model_20260924_ap_selected,RERANK_POOL=50"; j=$(sbatch --parsable --export="ALL,ARTICLE=$a,MODEL=Llama-3.3-70B-Instruct,RETRIEVER=tgn_kg,K=$k,TEXT=1,SPLIT=test,MAX_TOKENS=1200,MAX_MODEL_LEN=9600,ENDPOINT=$u$x" scripts/predict_retrieval.sh); echo "submitted k=$k rr=$rr article=$a job=$j"; done; }
for t in $(seq 1 48); do
 echo "===== $(date --iso-8601=seconds) tick $t/48 ====="; squeue -u "$USER" -o '%.10i %.24j %.10T %.10M %.25R' || true; n=$(squeue -h -u "$USER" -n predict_rag -t R,PD -o '%i' | wc -l | tr -d ' '); u=$(ep_ok || true); st=$(cat "$S" 2>/dev/null || echo k10); echo "stage=$st active_or_pending=$n endpoint=${u:-none}"
 if [ "$n" -eq 0 ] && [ -n "$u" ]; then case "$st" in
  k10) 
   if [ "$(complete 3 0)" = yes ] && [ "$(complete 5 0)" = yes ]; then submit 10 0 "$u"; echo rr3 > "$S";
   elif [ -f data/results/article3/retrieval_tgn_kg_k5_text1_test_Llama-3.3-70B-Instruct.jsonl ] && [ ! -f "$S.k5repair" ]; then
    echo "k5 has incomplete/null rows; scheduling one clean repair pass"; submit 5 0 "$u"; touch "$S.k5repair";
   fi;;
  rr3) [ "$(complete 10 0)" = yes ] && { submit 3 1 "$u"; echo rr5 > "$S"; };;
  rr5) [ "$(complete 3 1)" = yes ] && { submit 5 1 "$u"; echo rr10 > "$S"; };;
  rr10) [ "$(complete 5 1)" = yes ] && { submit 10 1 "$u"; echo done > "$S"; };;
 esac; fi
 [ "$t" -lt 48 ] && sleep 3600
done
