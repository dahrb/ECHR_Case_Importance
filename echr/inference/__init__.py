"""
echr.inference — LLM batch inference for all ablation conditions.

Responsibilities
----------------
- Load and run open-weight models (GPT-OSS, Llama 3.3, Qwen2.5-72B) locally on cluster
- Generate JSONL batch files of prediction prompts per experiment condition
- Submit and retrieve results from SLURM batch jobs
- Parse model output JSON → structured prediction records

Supported models
----------------
  GPT-OSS      — open-weights variant; local HuggingFace inference
  Llama-3.3    — meta-llama/Llama-3.3-*; uses vLLM or HF generate
  Qwen2.5-72B  — Qwen/Qwen2.5-72B-Instruct (already in OS_LLM_exp.py)

Inputs
------
  data/processed/article{N}/splits/comm_test.pkl    — test cases
  data/processed/article{N}/outcome_summaries.pkl   — 200/500-word summaries
  data/retrieval/article{N}/*_results.pkl           — pre-computed top-K (FEW_SHOT/RETRIEVAL)
  data/graph/article{N}/best_model.pth              — TGN model (KG condition)

Outputs
-------
  results/batches/article{N}/{model}/{condition}/predictions_{K}_{embedding}.jsonl
  results/batches/article{N}/{model}/{condition}/final_examples_{K}_{embedding}.pkl

Current implementation:
  old/Llama-3/OS_LLM_exp.py              — main inference loop (all conditions)
  old/PREDICTION/generate_pred_batch.py  — batch JSONL generator (OpenAI API version)
  old/Llama-3/finetune_script.py         — finetuning pipeline
Migration: refactor condition logic into this module; keep SLURM scripts in scripts/.
"""
