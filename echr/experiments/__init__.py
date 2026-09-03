"""
echr.experiments — Prompt templates and experiment condition logic.

Ablation conditions
-------------------
  BASE        Zero-shot prediction from law/fact sections alone
  COURT       Predict per court level (CHAMBER, COMMITTEE, etc.) then aggregate
  GOLD        Oracle: prior importance labels from other court levels provided as context
  FEW_SHOT    K retrieved similar cases (with importance labels) as in-context examples
  FT          Finetuned model variant — same prompt as BASE but different model weights
  RETRIEVAL   FEW_SHOT + BERT cross-encoder relevance filter (echr.rerank)
  CoT         Chain-of-thought: model reasons step-by-step before giving importance label
  KG          RETRIEVAL + TGN graph similarity to reorder/augment the example candidates

Prompt schema
-------------
  Output JSON: {"Case Importance": "key_case|1|2|3", "Reasoning": "..."}
  Input text: subject_matter (comm phase) or facts+law (judgment phase)

Current implementation:
  old/GPT_Experiments.py     — Experiment_2 class: get_rag_prompt(), prompt assembly
  old/PREDICTION/generate_pred_batch.py — condition routing + parameter grid
Migration: move Experiment_2 and prompt builders here; parametrise by article and model.
"""
