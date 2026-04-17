"""
Generic outcome-case (judgment) processor.

Usage:
    python Data_Process/judgment_cases.py --article_dir articles/art_3
    python Data_Process/judgment_cases.py --article_dir articles/art_6
    python Data_Process/judgment_cases.py --article_dir articles/art_8

Reads:
    <article_dir>/corpora/outcome/  (ECHR-OD corpus, pre-existing)
      Each branch subdirectory (ADMISSIBILITY, CHAMBER, …) contains:
        fact_section/  and  law_section/
    <article_dir>/raw_metadata/  (JSON files from s1)
    articles/key_labels.json

Writes:
    <article_dir>/outcome_cases.pkl
    <article_dir>/outcome_cases_for_network.pkl

Version history:
v1_0 = generic version of Art_3_Data_Process/judgment_cases.py; accepts --article_dir.
"""

import os
import sys
import importlib.util
import pandas as pd
from optparse import OptionParser

parser = OptionParser(usage='usage: %prog --article_dir <path>')
parser.add_option('--article_dir', dest='article_dir',
                  default='articles/art_3',
                  help='Path to the article directory (e.g. articles/art_3)')
(options, _) = parser.parse_args()

ARTICLE_DIR = options.article_dir

# ---------------------------------------------------------------------------
# Load article config
# ---------------------------------------------------------------------------
config_path = os.path.join(ARTICLE_DIR, 'config.py')
spec = importlib.util.spec_from_file_location('article_config', config_path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

print(f'Processing outcome cases for {cfg.ARTICLE_NAME}')

# ---------------------------------------------------------------------------
# Load text from judgment corpus
# ---------------------------------------------------------------------------
corpus_dir = os.path.join(ARTICLE_DIR, 'corpora', 'outcome')

facts_list = []
law_list   = []

for branch in os.listdir(corpus_dir):
    branch_path = os.path.join(corpus_dir, branch)
    if not os.path.isdir(branch_path):
        continue
    for section in ['fact_section', 'law_section']:
        section_path = os.path.join(branch_path, section)
        if not os.path.exists(section_path):
            continue
        for fname in os.listdir(section_path):
            fpath = os.path.join(section_path, fname)
            with open(fpath, 'r', encoding='utf-8') as fh:
                text = fh.read()
            entry = [fname, text]
            if section == 'fact_section':
                facts_list.append(entry)
            else:
                law_list.append(entry)

facts_df = pd.DataFrame(facts_list, columns=['Filename', 'Facts'])
law_df   = pd.DataFrame(law_list,   columns=['Filename', 'The Law'])

for df in [facts_df, law_df]:
    df['Filename'] = df['Filename'].str.replace('.txt', '', regex=False)

facts_df['Facts']    = facts_df['Facts'].str.strip().str.replace('\n', ' ', regex=False)
law_df['The Law']    = law_df['The Law'].str.strip().str.replace('\n', ' ', regex=False)
facts_df['Word Count'] = facts_df['Facts'].str.split().str.len()

df_merged = pd.merge(facts_df, law_df, on='Filename', how='inner')
df_merged = df_merged[df_merged['Word Count'] >= 60]

# Extract date and file ID from filename  (expected format: YYYY-MM-DD_ITEMID)
df_merged['date'] = df_merged['Filename'].str.split('_').str[0]
df_merged['File'] = df_merged['Filename'].str.split('_').str[1]
df_merged['date'] = pd.to_datetime(df_merged['date'], format='%Y-%m-%d', errors='coerce')
df_merged = df_merged[df_merged['date'] >= '1995-01-01']

# ---------------------------------------------------------------------------
# Load metadata from raw_metadata (s1 output)
# Falls back to repo-root raw_case_metadata/ if article-level dir is empty
# ---------------------------------------------------------------------------
raw_meta_dir = os.path.join(ARTICLE_DIR, 'raw_metadata')
if not any(f.endswith('.json') for f in os.listdir(raw_meta_dir) if os.path.exists(raw_meta_dir)):
    raw_meta_dir = 'raw_case_metadata'
metadata_frames = []
for fname in os.listdir(raw_meta_dir):
    if fname.endswith('.json'):
        fpath = os.path.join(raw_meta_dir, fname)
        metadata_frames.append(pd.read_json(fpath, lines=True))

if not metadata_frames:
    raise FileNotFoundError(f'No JSON metadata files found in {raw_meta_dir}')

metadata = pd.concat(metadata_frames, ignore_index=True)

cols = ['itemid', 'appno', 'doctypebranch', 'respondent', 'decisiondate',
        'extractedappno', 'conclusion', 'importance', 'kpthesaurus',
        'judgementdate', 'sclappnos']
metadata = metadata[[c for c in cols if c in metadata.columns]]

metadata['judgementdate'] = pd.to_datetime(metadata.get('judgementdate'), errors='coerce')
metadata['decisiondate']  = pd.to_datetime(metadata.get('decisiondate'),  errors='coerce')
metadata['judgementdate'] = metadata['judgementdate'].fillna(metadata['decisiondate'])
metadata.drop(columns=['decisiondate'], errors='ignore', inplace=True)
metadata.rename(columns={'itemid': 'File'}, inplace=True)
metadata.drop_duplicates(subset='File', keep=False, inplace=True)
metadata = metadata[metadata['judgementdate'] >= '1995-01-01']

full_data = pd.merge(df_merged, metadata, on='File', how='inner')
full_data.drop(columns=['Filename', 'judgementdate'], errors='ignore', inplace=True)

# ---------------------------------------------------------------------------
# Filter to article-specific keywords
# ---------------------------------------------------------------------------
article_keywords = cfg.KEYWORDS

full_data = full_data[full_data['kpthesaurus'].fillna('').apply(
    lambda x: any(k.strip() in article_keywords for k in x.split(';')))]

print(f'Outcome cases after filtering: {len(full_data)}')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(ARTICLE_DIR, 'outcome_cases.pkl')
full_data.to_pickle(out_path)
print(f'Saved outcome_cases.pkl → {out_path}')

# Network version: add 'date' as string for KG processing
net = full_data.copy()
net['date'] = net['date'].dt.strftime('%Y-%m-%d')
net_path = os.path.join(ARTICLE_DIR, 'outcome_cases_for_network.pkl')
net.to_pickle(net_path)
print(f'Saved outcome_cases_for_network.pkl → {net_path}')
