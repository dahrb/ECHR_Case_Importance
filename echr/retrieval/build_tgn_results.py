"""Build temporally safe, VDB-seeded TGN retrieval results.

This restores the original KG experiment: VDB results provide seed cases;
only citation events and candidate nodes strictly before each query date are
used; and candidates are ranked by cumulative cosine similarity to the seeds.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import pandas as pd
import torch

from echr.graph.data_process_KG import DATA


DEFAULT_SEED_KS = (3, 5, 10)
REPLAY_BATCH_SIZE = 5  # Match TGN training and the historical get_graph_at_time().


def first_appno(value) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    value = str(value).split(";")[0].strip()
    return value or None


def build_outcome_indexes(outcomes: pd.DataFrame):
    """Return appno->dated rows and filename->appno mappings."""
    by_appno = defaultdict(list)
    file_to_appno = {}
    for _, row in outcomes.iterrows():
        filename = row.get("File", row.get("Filename"))
        appno = first_appno(row.get("appno", row.get("Application Number")))
        date = pd.to_datetime(row.get("date", row.get("doc_date")), errors="coerce")
        if filename and appno and pd.notna(date):
            by_appno[appno].append((date, filename))
            file_to_appno[filename] = appno
    for rows in by_appno.values():
        rows.sort(key=lambda item: item[0])
    return by_appno, file_to_appno


def resolve_seed_ids(
    retrieved_appnos,
    query_date,
    by_appno,
    filename_to_int,
    active_ids,
    limit,
):
    """Resolve ranked VDB appnos to distinct graph nodes available at query time."""
    seeds = []
    seen = set()
    for raw in retrieved_appnos:
        appno = first_appno(raw)
        if not appno:
            continue
        eligible = [item for item in by_appno.get(appno, ()) if item[0] < query_date]
        if not eligible:
            continue
        filename = eligible[-1][1]
        node_id = filename_to_int.get(filename)
        if node_id is None or node_id not in active_ids or node_id in seen:
            continue
        seeds.append(node_id)
        seen.add(node_id)
        if len(seeds) == limit:
            break
    return seeds


@torch.no_grad()
def active_embeddings(model, active_ids):
    """Compute graph-attention embeddings for every currently active node."""
    ids = torch.tensor(sorted(active_ids), dtype=torch.long, device=model.data.src.device)
    n_id, edge_index, e_id = model.neighbor_loader(ids)
    model.assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
    memory, last_update = model.memory(n_id)
    z = model.gnn(
        memory,
        model.node_features[n_id - 1],
        last_update,
        edge_index,
        model.data.t[e_id],
        model.data.msg[e_id],
    )
    return ids, z[model.assoc[ids]]


def cumulative_rank(ids, embeddings, seed_ids, keep, allowed_ids=None):
    """Rank active node IDs by summed cosine similarity to seed IDs."""
    if not seed_ids:
        return []
    id_list = ids.tolist()
    positions = {node_id: pos for pos, node_id in enumerate(id_list)}
    usable_seeds = [node_id for node_id in seed_ids if node_id in positions]
    if not usable_seeds:
        return []

    z = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-12)
    seed_pos = torch.tensor([positions[node_id] for node_id in usable_seeds], device=z.device)
    scores = (z @ z[seed_pos].T).sum(dim=1)
    for node_id in usable_seeds:
        scores[positions[node_id]] = -torch.inf
    if allowed_ids is not None:
        allowed_positions = {positions[node_id] for node_id in allowed_ids if node_id in positions}
        if not allowed_positions:
            return []
        blocked = [pos for pos in range(len(id_list)) if pos not in allowed_positions]
        if blocked:
            scores[torch.tensor(blocked, device=z.device)] = -torch.inf

    n_keep = min(keep, len(id_list) - len(usable_seeds))
    if n_keep <= 0:
        return []
    top = torch.topk(scores, k=n_keep).indices.tolist()
    return [id_list[pos] for pos in top if torch.isfinite(scores[pos])]


def load_model(article: str, node_features: bool):
    # Import lazily so helper tests do not require CUDA.
    import echr.graph.network_TGN as tgn

    tgn.NODE_FEATURES_INC = node_features
    tgn.IN_CHANNELS = tgn.memory_dim
    tgn.set_seed(42)
    model = tgn.Model(article=article)
    model.load_data()

    checkpoint = os.path.join(
        DATA, "network", f"article{article}", f"best_model_{node_features}.pth"
    )
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Missing TGN checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=tgn.device, weights_only=False)
    model.load_parameters(state)
    model.memory.eval()
    model.gnn.eval()
    model.link_pred.eval()
    model.memory.reset_state()
    model.neighbor_loader.reset_state()
    return model


@torch.no_grad()
def build_tgn(
    article: str,
    seed_results_path: str | None = None,
    seed_ks=DEFAULT_SEED_KS,
    keep: int = 100,
    node_features: bool = False,
):
    vectordb_dir = os.path.join(DATA, "vectordb", f"article{article}")
    if seed_results_path is None:
        seed_results_path = os.path.join(
            vectordb_dir, "cosine_qwen3-8b_raw_chunk_2048_results.pkl"
        )

    comm = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "splits", "comm_test.pkl")
    ).copy()
    outcomes = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "outcome_cases.pkl")
    )
    seed_results = pd.read_pickle(seed_results_path)
    by_appno, _ = build_outcome_indexes(outcomes)

    model = load_model(article, node_features=node_features)
    filename_to_int = model.dataset.filename_to_int
    int_to_filename = {value: key for key, value in filename_to_int.items()}
    node_dates = {
        filename_to_int[filename]: date
        for appno_rows in by_appno.values()
        for date, filename in appno_rows
        if filename in filename_to_int
    }

    edges = pd.read_csv(
        os.path.join(DATA, "network", f"article{article}", "network_edges.csv"),
        usecols=["source"],
    )
    nodes = pd.read_csv(
        os.path.join(DATA, "network", f"article{article}", "network_nodes.csv"),
        usecols=["id", "date"],
    )
    reference_date = pd.to_datetime(
        edges.merge(nodes, left_on="source", right_on="id")["date"], errors="coerce"
    ).min()

    comm["_query_date"] = pd.to_datetime(comm["doc_date"], errors="coerce")
    comm = comm.dropna(subset=["_query_date"]).sort_values("_query_date")
    seed_ks = tuple(sorted(set(int(k) for k in seed_ks)))
    max_seed_k = max(seed_ks)
    results = {k: {} for k in seed_ks}

    event_times = model.data.t
    event_pos = 0
    num_events = model.data.num_events
    active_ids = set()
    missing_seed_results = 0
    no_seed_counts = {k: 0 for k in seed_ks}

    for query_no, (_, row) in enumerate(comm.iterrows(), start=1):
        query_date = row["_query_date"]
        query_t = int((query_date - reference_date).days)

        # Advance state using only events strictly before this query date.
        next_pos = int(torch.searchsorted(event_times, query_t, right=False).item())
        while event_pos < next_pos:
            # TGNMemory's LastAggregator retains one message per node within a
            # call, so replay batch size is part of the model semantics.  The
            # original implementation and training both used batches of five.
            end = min(event_pos + REPLAY_BATCH_SIZE, next_pos)
            src = model.data.src[event_pos:end]
            dst = model.data.dst[event_pos:end]
            t = model.data.t[event_pos:end]
            msg = model.data.msg[event_pos:end]
            model.memory.update_state(src, dst, t, msg)
            model.neighbor_loader.insert(src, dst)
            active_ids.update(src.tolist())
            active_ids.update(dst.tolist())
            event_pos = end

        filename = row["Filename"]
        retrieved = seed_results.get(filename)
        if not retrieved:
            missing_seed_results += 1
            for k in seed_ks:
                results[k][filename] = []
            continue

        seeds = resolve_seed_ids(
            retrieved,
            query_date,
            by_appno,
            filename_to_int,
            active_ids,
            max_seed_k,
        )
        if not active_ids or not seeds:
            for k in seed_ks:
                results[k][filename] = []
                no_seed_counts[k] += 1
            continue

        ids, embeddings = active_embeddings(model, active_ids)
        # A graph appearance is not sufficient evidence that an outcome
        # document was available: a small number of raw citation rows have a
        # same-day or forward-dated target.  Never return such a node.
        allowed_ids = {
            node_id for node_id in active_ids
            if node_dates.get(node_id, pd.Timestamp.max) < query_date
        }
        for k in seed_ks:
            selected_seeds = seeds[:k]
            if not selected_seeds:
                results[k][filename] = []
                no_seed_counts[k] += 1
                continue
            ranked_ids = cumulative_rank(
                ids, embeddings, selected_seeds, keep, allowed_ids=allowed_ids
            )
            ranked_files = []
            seen_files = set()
            for node_id in ranked_ids:
                ranked_file = int_to_filename.get(node_id)
                if ranked_file and ranked_file not in seen_files:
                    ranked_files.append(ranked_file)
                    seen_files.add(ranked_file)
                if len(ranked_files) == keep:
                    break
            results[k][filename] = ranked_files

        if query_no % 100 == 0:
            print(
                f"[art {article}] {query_no}/{len(comm)} queries; "
                f"events={event_pos}/{num_events}; active_nodes={len(active_ids)}",
                flush=True,
            )

    os.makedirs(vectordb_dir, exist_ok=True)
    for k, values in results.items():
        out_path = os.path.join(vectordb_dir, f"tgn_kg_k{k}_results.pkl")
        pd.to_pickle(values, out_path)
        nonempty = sum(bool(value) for value in values.values())
        print(f"[art {article}] k={k}: {nonempty}/{len(values)} non-empty -> {out_path}")
    print(
        f"[art {article}] missing VDB queries={missing_seed_results}; "
        f"no-seed counts={no_seed_counts}",
        flush=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="3, 6, or 8")
    parser.add_argument("--seed-results", default=None)
    parser.add_argument("--seed-k", type=int, nargs="+", default=list(DEFAULT_SEED_KS))
    parser.add_argument("--keep", type=int, default=100)
    parser.add_argument(
        "--node-features",
        action="store_true",
        help="Use feature-enabled checkpoint (original KG setup leaves this off)",
    )
    args = parser.parse_args()
    build_tgn(
        args.article,
        seed_results_path=args.seed_results,
        seed_ks=args.seed_k,
        keep=args.keep,
        node_features=args.node_features,
    )


if __name__ == "__main__":
    main()
