"""Fail-fast audit for temporally truncated KG retrieval artifacts."""

import argparse
import os

import pandas as pd

from echr.graph.data_process_KG import DATA
from echr.retrieval.build_tgn_results import build_outcome_indexes, first_appno


def audit(article: str, seed_ks=(3, 5, 10)):
    base = os.path.join(DATA, "processed", f"article{article}")
    vdb = os.path.join(DATA, "vectordb", f"article{article}")
    comm = pd.read_pickle(os.path.join(base, "splits", "comm_test.pkl"))
    outcomes = pd.read_pickle(os.path.join(base, "outcome_cases.pkl"))
    summaries = pd.read_pickle(os.path.join(base, "outcome_summaries.pkl"))
    seeds = pd.read_pickle(os.path.join(vdb, "cosine_qwen3-8b_raw_chunk_2048_results.pkl"))

    query_dates = {
        row["Filename"]: pd.to_datetime(row["doc_date"])
        for _, row in comm.iterrows()
    }
    outcome_dates = {
        row["File"]: pd.to_datetime(row["date"])
        for _, row in outcomes.iterrows()
    }
    summary_files = set(summaries["Filename"])
    by_appno, _ = build_outcome_indexes(outcomes)
    expected_keys = set(query_dates)

    network_dir = os.path.join(DATA, "network", f"article{article}")
    graph_nodes = pd.read_csv(
        os.path.join(network_dir, "network_nodes.csv"), usecols=["id", "date"]
    )
    graph_edges = pd.read_csv(
        os.path.join(network_dir, "network_edges.csv"), usecols=["source", "target"]
    )
    edge_dates = graph_edges.merge(
        graph_nodes, left_on="source", right_on="id", how="left"
    )
    first_event = {}
    for _, edge in edge_dates.iterrows():
        date = pd.to_datetime(edge["date"])
        for node in (edge["source"], edge["target"]):
            first_event[node] = min(first_event.get(node, date), date)

    reports = {}
    for k in seed_ks:
        path = os.path.join(vdb, f"tgn_kg_k{k}_results.pkl")
        results = pd.read_pickle(path)
        if set(results) != expected_keys:
            missing = expected_keys - set(results)
            extra = set(results) - expected_keys
            raise AssertionError(f"k={k} key mismatch: missing={len(missing)} extra={len(extra)}")

        future = []
        unknown = []
        unresolved_summary = []
        seed_leaks = []
        nonempty = 0
        for query_file, ranked_files in results.items():
            query_date = query_dates[query_file]
            if ranked_files:
                nonempty += 1

            resolved_seed_files = set()
            for raw in seeds.get(query_file, []):
                appno = first_appno(raw)
                eligible = [item for item in by_appno.get(appno, ()) if item[0] < query_date]
                if eligible:
                    seed_file = eligible[-1][1]
                    if first_event.get(seed_file, query_date) < query_date:
                        resolved_seed_files.add(seed_file)
                if len(resolved_seed_files) == k:
                    break

            for candidate in ranked_files:
                if candidate not in outcome_dates:
                    unknown.append((query_file, candidate))
                    continue
                if outcome_dates[candidate] >= query_date:
                    future.append((query_file, candidate, outcome_dates[candidate], query_date))
                if first_event.get(candidate, query_date) >= query_date:
                    future.append((query_file, candidate, "not active", query_date))
                if candidate in resolved_seed_files:
                    seed_leaks.append((query_file, candidate))
                if candidate not in summary_files:
                    unresolved_summary.append((query_file, candidate))

        if future or unknown or seed_leaks:
            raise AssertionError(
                f"k={k}: future={len(future)} unknown={len(unknown)} "
                f"seed_leaks={len(seed_leaks)}"
            )
        reports[k] = {
            "queries": len(results),
            "nonempty": nonempty,
            "missing_summaries_in_ranked_100": len(unresolved_summary),
        }

    print(f"[art {article}] TEMPORAL KG AUDIT PASSED: {reports}", flush=True)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True)
    args = parser.parse_args()
    audit(args.article)
