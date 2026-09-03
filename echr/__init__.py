"""
echr — ECHR case importance prediction library.

Submodules
----------
echr.retrieval    Vector search: FAISS indexes + BM25 retriever
echr.graph        Knowledge graph: Temporal Graph Network (TGN) over citation edges
echr.inference    LLM inference: batch prediction for GPT-OSS, Llama 3.3, Qwen2.5
echr.summarize    Case summarization: 200/500-word summaries of judgment + comm texts
echr.rerank       Cross-encoder re-ranking: BERT relevance filter for few-shot candidates
echr.experiments  Experiment conditions: prompt templates for BASE/COURT/GOLD/FEW_SHOT/CoT/KG
echr.evaluate     Metrics and results: macro F1, per-class analysis, results tables

Current implementations live in old/ — see each submodule's __init__.py for the mapping.
Migration path: copy and refactor from old/ into the corresponding submodule as each is updated.
"""
