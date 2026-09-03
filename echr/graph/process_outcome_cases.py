"""
Converts outcome_cases.pkl -> outcome_cases_network.pkl, replicating the
processing in old/data_process_draft.ipynb (cells 102-138).

Steps:
  1. sclappnos fallback: fill missing extractedappno from sclappnos; merge
     any appnos present in sclappnos but absent from extractedappno
  2. Remove own appno(s) from citation list
  3. Normalise cited appnos to the first_app_no of the dataset case that
     owns them (handles multi-appno cases; vectorised via lookup)
  4. Filter to in-dataset citations only
  5. Deduplicate on unique_app_no (sort desc by date, keep='last')

Output adds columns: appno_list, unique_app_no, citations, citation_len,
new_list_length  — matching outcome_cases_network.pkl schema.

Usage:
    python echr/graph/process_outcome_cases.py --article 3
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")


def process(article):
    in_path = os.path.join(DATA, "processed", f"article{article}", "outcome_cases.pkl")
    out_path = os.path.join(DATA, "processed", f"article{article}", "outcome_cases_network.pkl")

    df = pd.read_pickle(in_path)
    print(f"[art {article}] loaded {len(df)} cases")

    # --- sclappnos fallback for missing extractedappno ---
    df.loc[df["extractedappno"].isnull(), "extractedappno"] = df["sclappnos"]

    # --- build raw citation lists and merge sclappnos differences ---
    df["sclappnos_list"] = df["sclappnos"].apply(
        lambda x: [a.strip() for a in str(x).split(";") if a.strip()] if pd.notnull(x) else []
    )
    df["extractedappno_list"] = df["extractedappno"].apply(
        lambda x: [a.strip() for a in str(x).split(";") if a.strip()] if pd.notnull(x) else []
    )
    differences = df.apply(
        lambda row: list(set(row["sclappnos_list"]) - set(row["extractedappno_list"])), axis=1
    )
    has_diff = differences.apply(lambda d: len([i for i in d if str(i).strip()]) > 0)
    df["extractedappno_list"] = df["extractedappno_list"].where(
        ~has_diff,
        df["extractedappno_list"] + differences,
    )
    df.drop(columns=["sclappnos_list"], inplace=True)

    # --- build appno_list and first_app_no (unique_app_no) ---
    df["appno_list"] = df["appno"].apply(
        lambda x: [a.strip() for a in str(x).split(";") if a.strip()]
    )
    df["unique_app_no"] = df["appno_list"].apply(lambda x: x[0] if x else None)

    # --- remove own appno(s) from citation candidates ---
    df["extractedappno_list"] = df.apply(
        lambda row: [a for a in row["extractedappno_list"] if a not in row["appno_list"]],
        axis=1,
    )

    # --- build lookup: any appno -> first_app_no of owning case ---
    appno_to_first = {}
    for _, row in df.iterrows():
        first = row["unique_app_no"]
        for a in row["appno_list"]:
            if a and a not in appno_to_first:
                appno_to_first[a] = first

    # --- normalise citations to first_app_no, filter to in-dataset only ---
    def normalise(cited_list):
        seen, result = set(), []
        for a in cited_list:
            canonical = appno_to_first.get(a)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result

    df["citations"] = df["extractedappno_list"].apply(normalise)
    df["citation_len"] = df["citations"].apply(len)
    df["new_list_length"] = df["citation_len"]

    df.drop(columns=["extractedappno_list"], inplace=True)

    # --- deduplicate on unique_app_no (sort desc by date -> keep='last' = earliest) ---
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date", ascending=False, inplace=True)
    before = len(df)
    df.drop_duplicates(subset="unique_app_no", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[art {article}] after dedup on unique_app_no: {len(df)} cases (removed {before - len(df)})")

    pd.to_pickle(df, out_path)
    print(f"[art {article}] saved -> {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", required=True)
    args = ap.parse_args()
    process(args.article)
