"""
data_collection.py — How to reproduce the data/ folder from scratch.

All data covers HUDOC case law up to 2024-06-13.
This cutoff is baked into s1 and propagates through the entire pipeline:
metadata → labels → comm-phase text → judgment text.

All collection scripts live in data/data_collection/.
Run from the repo root: python data/data_collection.py

Steps
-----
s1  Fetch HUDOC metadata for all document branches
        cwd: data/raw_case_metadata/
        output: {BRANCH}_meta.json  (7 files)

s2  Find communicated cases that overlap with judgment cases
        cwd: data/raw_case_metadata/
        output: data/overlap_cases/pruned_{BRANCH}_meta.json

s3  Extract importance labels
        cwd: data/
        output: data/important_labels.csv

s4  Scrape comm-phase text (subject matter + questions) — all articles
        cwd: data/
        output: data/corpora/communication_phase/{subject_matter,questions}/
        note: run WITHOUT --judgment-only; article arg is ignored for comm phase

s5  Build keyword label dictionary from HUDOC taxonomy HTML
        cwd: data/  (requires key_labels_taxonomy.html)
        output: data/key_labels.json
        note: only re-run if the HUDOC taxonomy changes; current file already exists

s6  Extract article-specific itemids from raw_case_metadata
s4  Scrape judgment text per article (--judgment-only skips comm phase)
        cwd: data/
        output: data/corpora/article{N}/{BRANCH}/{subject_matter,questions,fact_section,law_section}/
        repeat for each article: 3, 6, 8, ...

CLI Usage
---------
Run full pipeline (all steps, all articles):
    python data/data_collection.py

Run only specific steps:
    python data/data_collection.py --steps s4 s6

Run only for specific articles (affects s6 + judgment s4 only):
    python data/data_collection.py --articles 6 8

Run judgment text scrape for Art 6 and 8 only:
    python data/data_collection.py --steps s4_judgment --articles 6 8

Available step names: s1, s2, s3, s4_comm, s5, s6, s4_judgment
Shorthand aliases:    s4 → runs both s4_comm and s4_judgment
                      all → runs everything (default)
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
SCRIPTS = REPO / "data" / "data_collection"
DATA = REPO / "data"

ALL_STEPS = ["s1", "s2", "s3", "s4_comm", "s5", "s6", "s4_judgment"]
DEFAULT_ARTICLES = ["3", "6", "8"]


def run(script: str, *args, cwd: Path):
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n>>> {' '.join(cmd)}  (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def expand_steps(raw: list[str]) -> list[str]:
    expanded = []
    for s in raw:
        if s == "all":
            return ALL_STEPS
        elif s == "s4":
            expanded += ["s4_comm", "s4_judgment"]
        else:
            expanded.append(s)
    # preserve ordering from ALL_STEPS
    seen = set()
    return [s for s in ALL_STEPS if s in expanded and not (seen.add(s) or s in seen)]


def main():
    parser = argparse.ArgumentParser(description="ECHR data collection pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        metavar="STEP",
        help=(
            "Steps to run. Choices: s1 s2 s3 s4_comm s4_judgment s5 s6 "
            "(aliases: s4=both s4 steps, all=full pipeline). Default: all"
        ),
    )
    parser.add_argument(
        "--articles",
        nargs="+",
        default=DEFAULT_ARTICLES,
        metavar="N",
        help="Articles to process for s6 and s4_judgment steps. Default: 3 6 8",
    )
    args = parser.parse_args()

    steps = expand_steps(args.steps)
    articles = args.articles

    print(f"Steps:    {steps}")
    print(f"Articles: {articles}")

    if "s1" in steps:
        (DATA / "raw_case_metadata").mkdir(parents=True, exist_ok=True)
        run("s1_extract_meta_v1_1.py", cwd=DATA / "raw_case_metadata")

    if "s2" in steps:
        run("s2_overlap_cases_v1_0.py", cwd=DATA / "raw_case_metadata")

    if "s3" in steps:
        run("s3_get_labels_v1_1.py", cwd=DATA)

    if "s4_comm" in steps:
        # comm phase is article-agnostic; article arg is ignored inside the script
        run("s4_extract_text_v2_1.py", "3", cwd=DATA)

    # s5 — only re-run if the HUDOC taxonomy HTML has changed
    if "s5" in steps:
        run("s5_key_labels_dictionary_v1_0.py", cwd=DATA)

    for article in articles:
        if "s6" in steps:
            (DATA / "article_itemids").mkdir(parents=True, exist_ok=True)
            run("s6_article6_itemids_v1_0.py", article, cwd=DATA)
        if "s4_judgment" in steps:
            run("s4_extract_text_v2_1.py", article, "--judgment-only", cwd=DATA)


if __name__ == "__main__":
    main()
