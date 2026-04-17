"""
Version history
v1_0 = scrapes judgment full-text from HUDOC for all non-COMMUNICATEDCASES
  branches (GRANDCHAMBER, CHAMBER, COMMITTEE, ADMISSIBILITY, ADMISSIBILITYCOM).
  Extracts the fact section (THE FACTS … THE LAW) and law section (THE LAW …
  FOR THESE REASONS) and writes them to:
      <article_dir>/corpora/outcome/<branch>/fact_section/<date>_<itemid>.txt
      <article_dir>/corpora/outcome/<branch>/law_section/<date>_<itemid>.txt
  Mirrors the structure of Art_3_Data/corpora/article3/ produced originally.
  Reuses the HTML-parsing helpers from s4_extract_text_v1_2.py.
"""

import os
import sys
import json
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

# ---------------------------------------------------------------------------
# HTML parsing helpers (copied from s4_extract_text_v1_2.py)
# ---------------------------------------------------------------------------

def get_text_tag(case_html, strings, identity):
    """Return the first tag whose stripped text matches one of strings."""
    for tag in case_html.find_all(True):
        tag_text = tag.get_text(strip=True).upper()
        for s in strings:
            if tag_text == s.upper():
                return tag
    return None


def preprocess_html(case_html):
    """Remove table elements from the HTML in-place."""
    for table in case_html.find_all('table'):
        table.decompose()


def process_paragraph(paragraph):
    """Extract clean text from a <p> or <ol> tag, skipping footnote refs."""
    def extract_text(element):
        text = ''
        for child in element:
            if child.name == 'a' and 'ftnref' in child.get('href', ''):
                return text
            if isinstance(child, NavigableString):
                text += str(child)
            elif child.name == 'span':
                if child.get_text(strip=True) == '\xa0':
                    text += ' '
                else:
                    text += extract_text(child)
            else:
                text += extract_text(child)
        return text
    cleaned = ' '.join(extract_text(paragraph).split())
    return cleaned.strip()


def check_passage(case_html, identity, start_strings, end_strings):
    """Extract text between start_strings header and end_strings header."""
    start_tag = get_text_tag(case_html, start_strings, identity)
    if not start_tag:
        return ""
    end_tag = get_text_tag(case_html, end_strings, identity)

    html_text = ""
    current_tag = start_tag
    while current_tag:
        if end_tag:
            if current_tag == end_tag or (
                current_tag.text and end_tag.text in current_tag.text
            ):
                break
        html_text += str(current_tag)
        current_tag = current_tag.next_element

    soup = BeautifulSoup(html_text, 'html.parser')
    paragraphs = soup.find_all(['p', 'ol'])
    return "\n".join(process_paragraph(p) for p in paragraphs)


def write_section(section_type, identity, text, missing_list, output_dir):
    """Write a section text file; record identity in missing_list if empty."""
    if not text:
        missing_list.append(identity)
        return missing_list
    section_dir = os.path.join(output_dir, section_type)
    os.makedirs(section_dir, exist_ok=True)
    with open(os.path.join(section_dir, identity + ".txt"), 'w', encoding='utf-8') as fh:
        fh.write(text)
    return missing_list


def write_missing(label, missing_list, output_dir):
    if missing_list:
        fpath = os.path.join(output_dir, label + ".txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(fpath, 'w') as fh:
            for item in missing_list:
                fh.write(item + '\n')


# ---------------------------------------------------------------------------
# Section boundary strings for judgment documents
# ---------------------------------------------------------------------------
FACT_HEADERS = ["THE FACTS", "THE FACTS AND PROCEDURE", "THE CIRCUMSTANCES OF THE CASE"]
LAW_HEADERS  = ["THE LAW", "AS TO THE LAW", "IN LAW"]
END_HEADERS  = [
    "FOR THESE REASONS", "FOR THESE REASON",
    "OPERATIVE PROVISIONS", "OPERATIVE PROVISION",
    "DECIDES AS FOLLOWS", "DECLARES"
]

JUDGMENT_BRANCHES = {
    "GRANDCHAMBER", "CHAMBER", "COMMITTEE",
    "ADMISSIBILITY", "ADMISSIBILITYCOM"
}


# ---------------------------------------------------------------------------
# Main scraping function
# ---------------------------------------------------------------------------
def scrape_outcome(cases, article_dir):
    """
    cases: list of dicts with keys: itemid, branch, date  (date = YYYY-MM-DD str)
    article_dir: root article directory (e.g. articles/art_6)
    """
    # Group missing lists by branch
    missing_facts = {}
    missing_law   = {}

    for case in tqdm(cases, desc="Scraping outcome texts"):
        itemid = case['itemid']
        branch = case['branch']
        date   = case['date']

        identity   = f"{date}_{itemid}"
        output_dir = os.path.join(article_dir, 'corpora', 'outcome', branch)
        os.makedirs(output_dir, exist_ok=True)

        if branch not in missing_facts:
            missing_facts[branch] = []
            missing_law[branch]   = []

        url  = ("https://hudoc.echr.coe.int/app/conversion/docx/html/body"
                f"?library=ECHR&id={itemid}")
        try:
            page = requests.get(url, timeout=30)
        except Exception as e:
            print(f"  Request error for {itemid}: {e}")
            missing_facts[branch].append(identity)
            missing_law[branch].append(identity)
            continue

        case_html = BeautifulSoup(page.content, 'html.parser')
        preprocess_html(case_html)

        fact_text = check_passage(case_html, identity, FACT_HEADERS, LAW_HEADERS + END_HEADERS)
        law_text  = check_passage(case_html, identity, LAW_HEADERS,  END_HEADERS)

        missing_facts[branch] = write_section(
            'fact_section', identity, fact_text, missing_facts[branch], output_dir)
        missing_law[branch]   = write_section(
            'law_section',  identity, law_text,  missing_law[branch],   output_dir)

    # Write missing-file logs per branch
    for branch in missing_facts:
        branch_dir = os.path.join(article_dir, 'corpora', 'outcome', branch)
        write_missing(f"{branch}_fact_section_missing", missing_facts[branch], branch_dir)
        write_missing(f"{branch}_law_section_missing",  missing_law[branch],   branch_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from optparse import OptionParser
    import pandas as pd

    parser = OptionParser()
    parser.add_option('--article_dir', dest='article_dir', default=None,
                      help='Path to article directory (e.g. articles/art_6). '
                           'Reads importance_labels.csv and raw_metadata/ from here.')
    (options, _) = parser.parse_args()

    if not options.article_dir:
        print("ERROR: --article_dir is required")
        sys.exit(1)

    article_dir = options.article_dir

    # Read importance_labels.csv — contains itemid and source_file (branch)
    labels_csv = os.path.join(article_dir, 'importance_labels.csv')
    labels_df  = pd.read_csv(labels_csv)

    # Build itemid → (branch, date) map from raw metadata JSON files
    raw_meta_dir = os.path.join(article_dir, 'raw_metadata')
    if not os.path.isdir(raw_meta_dir) or not any(
        f.endswith('.json') for f in os.listdir(raw_meta_dir)
    ):
        raw_meta_dir = 'raw_case_metadata'
        print(f"Falling back to {raw_meta_dir}")

    meta_map = {}   # itemid → {'branch': str, 'date': str}
    for fname in os.listdir(raw_meta_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(raw_meta_dir, fname)
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                branch = rec.get('doctypebranch', '')
                if branch not in JUDGMENT_BRANCHES:
                    continue
                itemid = rec.get('itemid', '')
                if not itemid:
                    continue
                # Use judgementdate, fall back to decisiondate
                raw_date = rec.get('judgementdate') or rec.get('decisiondate') or ''
                # HUDOC format: "DD/MM/YYYY HH:MM:SS"
                date_str = ''
                if raw_date:
                    try:
                        parts = raw_date.strip().split()[0].split('/')
                        date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    except Exception:
                        date_str = ''
                meta_map[itemid] = {'branch': branch, 'date': date_str}

    # Build case list — only judgment branches
    cases = []
    skipped = 0
    for _, row in labels_df.iterrows():
        itemid = str(row['itemid'])
        source = str(row.get('source_file', ''))
        branch_from_source = source.replace('_meta.json', '').replace('pruned_', '')
        if branch_from_source not in JUDGMENT_BRANCHES:
            skipped += 1
            continue
        info = meta_map.get(itemid)
        if info is None:
            # Try to get branch from source_file; date unknown
            info = {'branch': branch_from_source, 'date': 'unknown'}
        cases.append({'itemid': itemid, 'branch': info['branch'], 'date': info['date']})

    print(f"Judgment cases to scrape: {len(cases)}  (skipped comm/unknown: {skipped})")
    scrape_outcome(cases, article_dir)
    print("Done.")
