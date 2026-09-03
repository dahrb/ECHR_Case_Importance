"""
echr.retrieval — Vector search over judgment case corpora.

Responsibilities
----------------
- Build and persist FAISS indexes (legal-bert-512, openai-512, openai-2048) per article
- Build and persist BM25 retriever per article
- Query: given a comm-phase test case, return top-K similar outcome cases
- MMR (maximal marginal relevance) re-ranking of FAISS results

Inputs
------
  data/processed/article{N}/outcome_cases.pkl   — judgment case corpus with Facts column

Outputs
-------
  data/retrieval/article{N}/faiss_{embedding}.bin
  data/retrieval/article{N}/bm25_retriever.pkl
  data/retrieval/article{N}/{embedding}_results.pkl  — pre-computed top-K per test case

Current implementation: old/VectorDB/initialise_vector_dbs.py
Migration: generalise to accept --article arg, move logic here.
"""
