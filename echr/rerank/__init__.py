"""
echr.rerank — BERT cross-encoder relevance filter for few-shot candidate selection.

Responsibilities
----------------
- Given a test case summary and a pool of candidate outcome cases, score each candidate
  for relevance using a fine-tuned BERT cross-encoder
- Filter candidates: keep only those scoring > 0.5 (relevant), then take top-K
- Used inside the RETRIEVAL and KG conditions to avoid irrelevant few-shot examples

Inputs
------
  Test case 200-word summary (from echr.summarize)
  Candidate case 200-word summaries (from outcome_summaries.pkl)
  Trained cross-encoder model weights

Outputs
-------
  Ranked + filtered list of (fileNo, importance) pairs of length K

Model
-----
  Fine-tuned sentence-transformers CrossEncoder
  Weights: old/BERT-rerank/model2024-11-07_14-39-42_FINAL/
  (to be migrated to: models/rerank/)

Current implementation:
  old/PREDICTION/generate_pred_batch.py — relevance_check() function (lines 127-141)
  old/BERT-rerank/                       — training scripts and saved model
Migration: extract relevance_check() and CrossEncoder loading into this module.
"""
