"""
Extract TGN node memory embeddings for comm_test cases and all outcome cases.

For each article, loads the trained TGN checkpoint, replays all citation edges
to warm up node memory, then extracts the 100-dim memory state for every outcome
case in the citation graph.  Results are saved to two files:

    data/network/article{N}/tgn_embeddings.pkl
        {comm_Filename: np.ndarray shape (100,)}
        Comm test cases with no matching outcome-case node get a zero vector.

    data/network/article{N}/outcome_tgn_embeddings.pkl
        {outcome_Filename: np.ndarray shape (100,)}
        All outcome cases in the citation graph (needed for TGN KNN retrieval).

Usage (via SLURM, see scripts/extract_tgn_embeddings.sh):
    python echr/graph/extract_tgn_embeddings.py --article 3
    python echr/graph/extract_tgn_embeddings.py --article 6
    python echr/graph/extract_tgn_embeddings.py --article 8
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import TemporalDataLoader

# Reuse the existing model components
from echr.graph.network_TGN import (
    Model,
    NODE_FEATURES_INC,
    memory_dim,
    time_dim,
    embedding_dim,
    NEIGHBORS,
    device,
    set_seed,
)
from echr.graph.data_process_KG import DATA

SEED = 42


def checkpoint_path(article: str) -> str:
    return os.path.join(DATA, "network", f"article{article}", f"best_model_True.pth")


def load_checkpoint(model: Model, article: str) -> None:
    path = checkpoint_path(article)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.gnn.load_state_dict(ckpt['model_state_dict'])
    model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    model.memory.load_state_dict(ckpt['memory_state_dict'])
    model.link_pred.load_state_dict(ckpt['link_pred_state_dict'])
    print(f"[art {article}] checkpoint loaded from {path}", flush=True)


@torch.no_grad()
def warm_up_memory(model: Model) -> None:
    """Replay all edges in temporal order to fill node memory states."""
    model.memory.eval()
    model.gnn.eval()
    model.neighbor_loader.reset_state()
    model.memory.reset_state()

    loader = TemporalDataLoader(model.data, batch_size=50, neg_sampling_ratio=0)
    for batch in loader:
        batch = batch.to(device)
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        model.neighbor_loader.insert(batch.src, batch.dst)
    print(f"[art {model.article}] memory warmed up over {model.data.num_events} events", flush=True)


@torch.no_grad()
def extract_all_memory(model: Model) -> dict:
    """Return {node_id (int): np.ndarray (memory_dim,)} for every graph node."""
    model.memory.eval()
    # Nodes are 1-indexed (1..N); TGNMemory is sized num_nodes = N+1 (PyG convention).
    # Valid node IDs: 1 to num_nodes-1 inclusive.
    num_nodes = model.data.num_nodes  # = N+1
    all_node_ids = torch.arange(1, num_nodes, device=device)  # [1, N]
    # TGNMemory forward: returns (memory, last_update) for requested node ids
    memory_states, _ = model.memory(all_node_ids)
    memory_np = memory_states.cpu().numpy()  # (N, memory_dim)
    node_id_to_emb = {int(nid): memory_np[i] for i, nid in enumerate(all_node_ids.tolist())}
    print(f"[art {model.article}] extracted memory for {len(node_id_to_emb)} nodes "
          f"(dim={memory_np.shape[1]})", flush=True)
    return node_id_to_emb


def map_to_comm_test(article: str, node_id_to_emb: dict, filename_to_int: dict) -> dict:
    """Map node embeddings to comm_test Filenames via shared appno."""
    comm_test = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "splits", "comm_test.pkl")
    )
    outcome_cases = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "outcome_cases.pkl")
    )

    # Build appno → outcome Filename map
    def _first_appno(val):
        if isinstance(val, list):
            return val[0] if val else None
        return str(val).split(';')[0].strip() if val else None

    appno_to_outcome_fn = {}
    for _, row in outcome_cases.iterrows():
        fn = row.get('Filename') or row.get('File')
        appno = _first_appno(row.get('appno') or row.get('Application Number'))
        if appno and fn:
            appno_to_outcome_fn[appno] = fn

    zero_vec = np.zeros(memory_dim, dtype=np.float32)
    results = {}
    matched = 0

    for _, row in comm_test.iterrows():
        comm_fn = row.get('Filename') or row.get('File')
        # Try to find matching outcome case via appno
        appno = _first_appno(row.get('appno') or row.get('Application Number'))
        outcome_fn = appno_to_outcome_fn.get(appno) if appno else None
        node_id = filename_to_int.get(outcome_fn) if outcome_fn else None
        if node_id and node_id in node_id_to_emb:
            results[comm_fn] = node_id_to_emb[node_id].astype(np.float32)
            matched += 1
        else:
            results[comm_fn] = zero_vec.copy()

    n_total = len(results)
    n_zero = n_total - matched
    print(f"[art {article}] comm_test mapping: {matched}/{n_total} matched, "
          f"{n_zero} zero vectors", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--article', required=True, help='3, 6, or 8')
    args = parser.parse_args()
    art = args.article

    set_seed(SEED)

    print(f"[art {art}] loading TGN model and data...", flush=True)
    # NODE_FEATURES_INC=True matches how checkpoints were trained
    import echr.graph.network_TGN as tgn_mod
    tgn_mod.NODE_FEATURES_INC = True
    tgn_mod.IN_CHANNELS = memory_dim  # will be corrected in load_data

    model = Model(article=art)
    model.load_data()
    load_checkpoint(model, art)

    warm_up_memory(model)

    node_id_to_emb = extract_all_memory(model)

    # filename_to_int is built in ECHRData.pre_process() and stored on the dataset
    filename_to_int = model.dataset.filename_to_int

    results = map_to_comm_test(art, node_id_to_emb, filename_to_int)

    out_path = os.path.join(DATA, "network", f"article{art}", "tgn_embeddings.pkl")
    pd.to_pickle(results, out_path)
    print(f"[art {art}] saved {len(results)} embeddings → {out_path}", flush=True)

    # Also save outcome-case embeddings (filename → embedding) for KNN retrieval
    int_to_filename = {v: k for k, v in filename_to_int.items()}
    outcome_embs = {
        int_to_filename[nid]: emb.astype(np.float32)
        for nid, emb in node_id_to_emb.items()
        if nid in int_to_filename
    }
    out_path2 = os.path.join(DATA, "network", f"article{art}", "outcome_tgn_embeddings.pkl")
    pd.to_pickle(outcome_embs, out_path2)
    print(f"[art {art}] saved {len(outcome_embs)} outcome embeddings → {out_path2}", flush=True)


if __name__ == '__main__':
    main()
