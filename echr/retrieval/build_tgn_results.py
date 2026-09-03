"""
Build TGN KNN retrieval results using TGN node memory embeddings.

For each comm_test case that has a non-zero TGN embedding (matched via appno to
an outcome case in the citation graph), finds the K most similar outcome cases
by cosine similarity in TGN memory space.

Requires: data/network/article{N}/outcome_tgn_embeddings.pkl (built by
          extract_tgn_embeddings.py --article N)
      and: data/network/article{N}/tgn_embeddings.pkl

Output: data/vectordb/article{N}/tgn_kg_results.pkl
    Format: {comm_Filename: [outcome_File, ...]} — same as bm25/faiss results.

Unmatched comm cases (zero vector) get an empty list (no examples).
"""

import argparse
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
KEEP = 100


def cosine_sim_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query vector and each row of matrix."""
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-8:
        return np.zeros(len(matrix))
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms = np.where(m_norms < 1e-8, 1e-8, m_norms)
    return (matrix @ query) / (m_norms * q_norm)


def build_tgn(article: str) -> None:
    net_dir = os.path.join(DATA, "network", f"article{article}")

    comm_emb_path = os.path.join(net_dir, "tgn_embeddings.pkl")
    outcome_emb_path = os.path.join(net_dir, "outcome_tgn_embeddings.pkl")

    if not os.path.exists(outcome_emb_path):
        raise FileNotFoundError(
            f"outcome_tgn_embeddings.pkl not found at {outcome_emb_path}. "
            "Run: python echr/graph/extract_tgn_embeddings.py --article {article}"
        )

    comm_embs: dict = pd.read_pickle(comm_emb_path)        # {comm_fn: np.array(100,)}
    outcome_embs: dict = pd.read_pickle(outcome_emb_path)  # {outcome_fn: np.array(100,)}

    outcome_files = list(outcome_embs.keys())
    outcome_matrix = np.stack([outcome_embs[f] for f in outcome_files])  # (N_out, 100)

    # Load outcome_cases to get importance (for summary existence filter later, but
    # for retrieval we just need the ranked filename list)
    results = {}
    matched = 0
    for comm_fn, emb in comm_embs.items():
        if np.linalg.norm(emb) < 1e-8:
            results[comm_fn] = []
            continue
        sims = cosine_sim_matrix(emb, outcome_matrix)
        top_idx = np.argsort(sims)[::-1][:KEEP]
        ranked = [outcome_files[i] for i in top_idx if sims[i] > 1e-8]
        results[comm_fn] = ranked
        matched += 1

    out_dir = os.path.join(DATA, "vectordb", f"article{article}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tgn_kg_results.pkl")
    pd.to_pickle(results, out_path)

    avg_k = sum(len(v) for v in results.values()) / len(results) if results else 0
    print(f"[art {article}] tgn_kg_results: {len(results)} cases, "
          f"{matched} with non-zero embeddings, avg {avg_k:.1f} docs/case")
    print(f"[art {article}] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="3, 6, or 8")
    args = parser.parse_args()
    build_tgn(args.article)


if __name__ == "__main__":
    main()
