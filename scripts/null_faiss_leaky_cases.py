"""
Identify FAISS cases where get_examples() would now return different context
(due to the query_date fix) and null their predictions so --resume reruns them.
Backs up each file before modifying.
"""
import json
import os
import shutil

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")


def get_affected_filenames(article, k):
    """Return set of Filenames whose FAISS context changes with the date fix."""
    comm = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "splits", "comm_test.pkl")
    )
    oc = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{article}", "outcome_cases.pkl")
    )
    faiss = pd.read_pickle(
        os.path.join(DATA, "vectordb", f"article{article}",
                     "cosine_qwen3-8b_raw_chunk_2048_results.pkl")
    )

    # appno -> latest outcome doc
    appno_latest = {}
    for _, row in oc.iterrows():
        key = str(row["appno"]).split(";")[0].strip()
        if key not in appno_latest or row["date"] > appno_latest[key]["date"]:
            appno_latest[key] = row

    # appno -> best pre-query-date outcome doc
    def best_before(appno_raw, qdate):
        first = str(appno_raw).split(";")[0].strip()
        rows = [r for r in [appno_latest.get(first)] if r is not None]
        # need full list from oc
        return rows

    # rebuild full appno_index (list of rows)
    appno_index = {}
    for _, row in oc.iterrows():
        key = str(row["appno"]).split(";")[0].strip()
        appno_index.setdefault(key, []).append(row)

    query_dates = dict(zip(comm["Filename"], pd.to_datetime(comm["doc_date"])))

    affected = set()
    for fname, retrieved in faiss.items():
        if fname not in query_dates:
            continue
        qdate = query_dates[fname]
        top_k = retrieved[:k]
        for appno_raw in top_k:
            first = str(appno_raw).split(";")[0].strip()
            rows = appno_index.get(first, [])
            if not rows:
                continue
            latest = sorted(rows, key=lambda r: r["date"])[-1]
            if latest["date"] >= qdate:
                # This appno's context changes with the fix
                affected.add(fname)
                break
    return affected


def null_predictions(path, affected_filenames, dry_run=False):
    """Rewrite file with prediction=None for affected cases. Returns (nulled, kept)."""
    if not os.path.exists(path):
        return 0, 0

    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    nulled = 0
    kept = 0
    new_lines = []
    for line in lines:
        try:
            record = json.loads(line)
        except Exception:
            new_lines.append(line)
            continue
        if record.get("Filename") in affected_filenames and record.get("prediction") is not None:
            record["prediction"] = None
            record["raw_prediction"] = None
            record["raw_output"] = None
            new_lines.append(json.dumps(record))
            nulled += 1
        else:
            new_lines.append(line)
            kept += 1

    if nulled > 0 and not dry_run:
        bak = path + ".pre_datefix.bak"
        shutil.copy2(path, bak)
        with open(path, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
        print(f"  Backed up to {os.path.basename(bak)}, nulled {nulled} predictions")
    elif nulled > 0:
        print(f"  [DRY RUN] would null {nulled} predictions in {os.path.basename(path)}")
    else:
        print(f"  No affected cases found in {os.path.basename(path)}")

    return nulled, kept


def main():
    endpoint = open(os.path.join(DATA, "vllm_gptoss_base_endpoint.txt")).read().strip()
    print(f"Base endpoint: {endpoint}\n")

    for article in [3, 6, 8]:
        res_dir = os.path.join(DATA, "results", f"article{article}")
        for k in [3, 5, 10]:
            affected = get_affected_filenames(article, k)
            print(f"Art{article} FAISS k={k}: {len(affected)} affected Filenames")

            for rerank in [False, True]:
                rr = "_rerank" if rerank else ""
                fname = f"retrieval_faiss_k{k}{rr}_text1_test_gpt-oss-120b.jsonl"
                path = os.path.join(res_dir, fname)
                null_predictions(path, affected)

        print()


if __name__ == "__main__":
    main()
