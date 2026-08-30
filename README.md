# ECHR Case Importance Prediction

Replication and extension of the ECHR case importance prediction pipeline across Articles 3, 6, and 8. Uses open-weight LLMs (GPT-OSS, Llama 3.3, Qwen2.5-72B) with ablation conditions including retrieval-augmented generation, knowledge graph integration, and finetuning.

## Repository structure

```
echr/                   # Python library — import and use pipeline modules
  retrieval/            #   FAISS + BM25 vector search
  graph/                #   Temporal Graph Network (TGN) over citation edges
  inference/            #   LLM batch inference (GPT-OSS, Llama 3.3, Qwen2.5)
  summarize/            #   200/500-word case summarization
  rerank/               #   BERT cross-encoder relevance filtering
  experiments/          #   Prompt templates for all ablation conditions
  evaluate/             #   Metrics, results tables, error analysis

data/                   # All pipeline data (see data/ layout below)
  data_collection.py    #   Pipeline orchestrator — run this to reproduce data/
  data_collection/      #   Collection scripts s1–s6 + preprocessing helpers
  raw_case_metadata/    #   s1: raw HUDOC metadata JSONs
  overlap_cases/        #   s2: communicated ↔ judgment overlap
  article_itemids/      #   s6: article-specific itemid lists
  corpora/              #   s4: scraped case text (fact_section, law_section, subject_matter)
  processed/            #   Built pkls: outcome_cases, comm_cases, splits per article
  network/              #   KG node/edge CSVs per article
  important_labels.csv  #   s3: importance labels (all articles)
  key_labels.json       #   s5: HUDOC keyword taxonomy

scripts/                # SLURM job scripts and CLI wrappers
notebooks/              # Analysis and exploration notebooks
docs/                   # Pipeline checklist and design docs
old/                    # Original Art 3 implementations (reference; not actively maintained)
```

## Setup

```bash
# Core dependencies only
uv sync

# Add optional groups as needed
uv sync --group nlp          # transformers, sentence-transformers
uv sync --group retrieval    # FAISS, LangChain
uv sync --group graph        # PyTorch + PyG (CPU)
uv sync --group graph-gpu    # PyTorch + PyG (CUDA 12.1)
uv sync --group notebooks    # Jupyter, matplotlib
```

## Reproducing the data

```bash
# Full pipeline from scratch (hours — run via SLURM)
python data/data_collection.py

# Specific steps / articles only
python data/data_collection.py --steps s4_judgment --articles 6 8
python data/data_collection.py --steps s6 s4_judgment --articles 6
```

## Ablation conditions

| Condition | Description |
|-----------|-------------|
| `BASE` | Zero-shot prediction from case text alone |
| `COURT` | Predict per court level, then aggregate |
| `GOLD` | Oracle: prior court-level labels as context |
| `FEW_SHOT` | K similar cases as in-context examples |
| `FT` | Finetuned model checkpoint |
| `RETRIEVAL` | FEW_SHOT + BERT cross-encoder relevance filter |
| `CoT` | Chain-of-thought reasoning prompt |
| `KG` | RETRIEVAL + TGN graph similarity reranking |

## Evaluation metric

**Macro F1** — primary metric. Accuracy is misleading: 81% of cases are importance level 4.

## Pipeline progress

See [docs/PIPELINE_CHECKLIST.md](docs/PIPELINE_CHECKLIST.md) for the full task breakdown and current status.
