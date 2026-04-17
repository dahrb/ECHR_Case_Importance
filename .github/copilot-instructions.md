# Copilot Instructions for ECHR_Importance

## Big picture (how this repo is organized)
- This is a research-style Python codebase with multiple independent pipelines, not a single packaged app.
- Main workflow is: HUDOC data extraction → overlap/label linking → text extraction/preprocessing → model/batch experiments → result scoring.
- Key data pipeline scripts are in `Data/`: run `s1_extract_meta_v1_1.py`, `s2_overlap_cases_v1_0.py`, `s3_get_importance_v1_1.py`, then `s4_extract_text_v1_2.py`.
- GPT experiment generation and prompt logic lives in `GPT_Experiments.py` and `Keyword_Prediction/keyword_prediction.py`.
- Batch upload/download integration with OpenAI is in `submit_batch_files.py`; metrics are computed in `results.py`.
- Retrieval experiments are under `VectorDB/` (`initialise_vector_dbs.py`, `run_experiments.py`) and persist FAISS/docstore artifacts.
- Graph experiments are under `Knowledge_Graph/` (notably `network_TGN.py`), which assumes CUDA availability.

## Project-specific conventions to preserve
- Keep existing “version history” docstrings at top of scripts; this repo tracks evolution there.
- Prefer extending existing scripts/classes over introducing new framework structure.
- Preserve file naming conventions with version suffixes (`*_v1_2.py`) and experiment-oriented names (`experiment_*`, `*_results.pkl`).
- Many scripts are designed as direct executables with hardcoded paths and an `if __name__ == '__main__':` block.
- Maintain key dataframe columns used across modules: `Filename`, `Subject Matter`, `Questions`, `importance`, `appno`, `docname`.
- Batch payload format is JSONL with `custom_id` + `/v1/chat/completions` body and JSON response format (see `Experiment_1.run_async`).

## Critical workflows and commands
- Environment is Conda-based (`environment.yml`, `conda-lock.yml`) and HPC jobs use `mamba activate ECHR` (`exp_1.sh`).
- Typical local run pattern:
  - `python data_preprocessing_COMM.py`
  - `python GPT_Experiments.py`
  - `python submit_batch_files.py`
  - `python results.py`
- Simple metric sanity test exists in `test_rmse.py` (`python test_rmse.py`).
- Vector retrieval runs are argument-driven, e.g. `python VectorDB/run_experiments.py -c 512 -o 50 -e nlpaueb/legal-bert-base-uncased -n legal-bert_raw -s cosine`.
- Knowledge graph training uses CLI flags in `Knowledge_Graph/network_TGN.py` (for example `--epochs`, `--n_runs`, `--node_features_inc`).

## Integration points and external dependencies
- HUDOC HTTP endpoints are queried directly in `Data/s1_extract_meta_v1_1.py` and case HTML conversion is scraped in `Data/s4_extract_text_v1_2.py`.
- OpenAI integrations use `openai.OpenAI` clients (`GPT_Experiments.py`, `submit_batch_files.py`, `Keyword_Prediction/keyword_prediction.py`).
- Retrieval stack combines FAISS + LangChain + HuggingFace/OpenAI embeddings under `VectorDB/`.
- Some scripts rely on `torch`, `torch_geometric`, and GPU-only execution paths (`Knowledge_Graph/network_TGN.py`).

## Agent guardrails for edits
- Do not refactor absolute-path usage broadly unless requested; many scripts assume `/users/sgdbareh/volatile/ECHR_Importance`.
- Keep output artifact formats and locations stable (`.jsonl`, `.pkl`, overlap JSONL, `importance_labels.csv`) to avoid breaking downstream scripts.
- When changing prompt/experiment code, preserve schema keys used by scoring (`Case Importance`, `Court`, etc.) in `results.py`.
- Treat `API_key.py` as local-secret plumbing; prefer environment-variable fallback for new code, and never add new hardcoded secrets.
- For data extraction changes, preserve section boundary behavior in `s4_extract_text_v1_2.py` (subject vs questions, footnote/table filtering).