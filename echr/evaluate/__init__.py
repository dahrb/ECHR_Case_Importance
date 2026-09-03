"""
echr.evaluate — Metrics, results aggregation, and analysis.

Responsibilities
----------------
- Parse raw prediction JSONL outputs → structured DataFrames
- Compute macro F1, per-class F1, accuracy per experiment condition
- Aggregate across articles, models, and conditions into summary tables
- Error analysis: breakdown by court level, importance level, article

Primary metric: macro F1 (accuracy is misleading — 81% of cases are importance 4)

Inputs
------
  results/batches/article{N}/{model}/{condition}/*.jsonl  — raw predictions
  data/processed/article{N}/splits/comm_test.pkl          — ground truth labels

Outputs
-------
  results/tables/summary_{article}_{model}.csv   — per-condition metrics
  results/tables/full_ablation.csv               — unified table across all experiments

Current implementation:
  old/results.py       — result parsing and scoring (Art 3 only, GPT-4o)
  old/Results/         — saved result files from original experiments
Migration: generalise results.py to handle any article/model; move here.
"""
