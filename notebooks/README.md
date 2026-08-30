# notebooks/

Exploratory analysis and one-off investigation notebooks.

## Naming convention

```
{phase}_{description}.ipynb
```

## Notebook index

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Case counts, importance distributions, corpus coverage per article |
| `02_corpus_validation.ipynb` | Verify fact/law section extraction completeness after s4 scrape |
| `03_label_analysis.ipynb` | Cross-reference important_labels.csv against article{N}_cases.json |
| `04_retrieval_analysis.ipynb` | Inspect top-K FAISS / BM25 retrieval quality |
| `05_kg_analysis.ipynb` | Citation graph structure, TGN training curves |
| `06_results_analysis.ipynb` | Ablation table, error analysis, per-class F1 breakdown |

## Legacy notebooks (in old/)

- `old/data_exploration.ipynb` — original Art 3 exploration
- `old/finetune_data_creation.ipynb` — GPT-4o finetuning data prep

These are kept in `old/` for reference; clean versions will be added here.
