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
        output: data/corpora/article{N}/{BRANCH}/{subject_matter,questions}/
        repeat for each article: 3, 6, 8, ...
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
SCRIPTS = REPO / "data" / "data_collection"
DATA = REPO / "data"


def run(script: str, *args, cwd: Path):
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n>>> {' '.join(cmd)}  (cwd={cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


if __name__ == "__main__":
    # s1 — metadata
    (DATA / "raw_case_metadata").mkdir(parents=True, exist_ok=True)
    run("s1_extract_meta_v1_1.py", cwd=DATA / "raw_case_metadata")

    # s2 — overlap
    run("s2_overlap_cases_v1_0.py", cwd=DATA / "raw_case_metadata")

    # s3 — labels
    run("s3_get_labels_v1_1.py", cwd=DATA)

    # s4 — comm-phase text (all articles; no article arg needed)
    run("s4_extract_text_v2_1.py", "3", cwd=DATA)

    # s5 — keyword dictionary (uncomment if taxonomy HTML has changed)
    # run("s5_key_labels_dictionary_v1_0.py", cwd=DATA)

    # s6 + s4 — judgment text per article
    for article in ["3", "6", "8"]:
        run("s6_article6_itemids_v1_0.py", article, cwd=DATA)
        run("s4_extract_text_v2_1.py", article, "--judgment-only", cwd=DATA)
