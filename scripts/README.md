# scripts/

SLURM job scripts and CLI wrappers for running pipeline stages on the cluster.

## Naming convention

```
{stage}_{description}.sh     — SLURM batch scripts
{stage}_{description}.py     — Python CLI entry points (thin wrappers over echr.*)
```

## Stages

| Script | Purpose |
|--------|---------|
| `collect_*.sh` | Run `data/data_collection.py` steps via SLURM |
| `build_retrieval_*.sh` | Build FAISS + BM25 indexes for a given article |
| `build_graph_*.sh` | Train TGN knowledge graph model for a given article |
| `summarize_*.sh` | Generate case summaries via GPT-OSS |
| `run_inference_*.sh` | Submit LLM inference jobs for a given model × article × condition |
| `finetune_*.sh` | Finetuning jobs (GPT-OSS / Llama 3.3 / Qwen2.5) |

## Current SLURM scripts (in old/ or data/data_collection/)

- `data/data_collection/pipeline_art6.sh` — s4 + s6 for Art 6
- `data/data_collection/pipeline_art8.sh` — s4 + s6 for Art 8
- `data/data_collection/pipeline_multi_article.sh` — combined Art 6 + 8
- `old/Llama-3/llamaMultiNODE*.sh` — multi-node inference for Llama

These will be migrated here as each pipeline stage is refactored into `echr/`.

## Usage

All scripts expect to be submitted from the repo root:

```bash
sbatch scripts/build_retrieval_art6.sh
sbatch scripts/run_inference_gpt_oss_art6_base.sh
```

Set `ECHR_ROOT=/mnt/scratch/users/sgdbareh/ECHR_Importance` in your environment or
each script's `#SBATCH --chdir` directive.
