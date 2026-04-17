"""
Generic communicated-case processor.

Usage:
    python Data_Process/comm_cases.py --article_dir articles/art_3
    python Data_Process/comm_cases.py --article_dir articles/art_6
    python Data_Process/comm_cases.py --article_dir articles/art_8

Reads:
    <article_dir>/corpora/communication_phase/questions/
    <article_dir>/corpora/communication_phase/subject_matter/
    <article_dir>/importance_labels.csv
    <article_dir>/overlap_cases/COMMUNICATEDCASES_meta.json  (for kpthesaurus)
    articles/key_labels.json

Writes:
    <article_dir>/comm_cases.pkl
    <article_dir>/comm_cases_valid.pkl   (sampled validation set)
    <article_dir>/comm_cases_test.pkl

Version history:
v1_0 = generic version of Art_3_Data_Process/comm_cases.py; accepts --article_dir.
"""

import os
import sys
import json
import importlib.util
import pandas as pd
from optparse import OptionParser

# Allow importing data_preprocessing_COMM from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import data_preprocessing_COMM as dpc

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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

print(f'Processing communicated cases for {cfg.ARTICLE_NAME}')

# ---------------------------------------------------------------------------
# Load thesaurus
# ---------------------------------------------------------------------------
key_labels_path = os.path.join(os.path.dirname(ARTICLE_DIR), 'key_labels.json')
with open(key_labels_path) as f:
    label_to_keyword = json.load(f)

# ---------------------------------------------------------------------------
# Load comm case text
# ---------------------------------------------------------------------------
questions_dir   = os.path.join(ARTICLE_DIR, 'corpora', 'communication_phase', 'questions', '')
subject_dir     = os.path.join(ARTICLE_DIR, 'corpora', 'communication_phase', 'subject_matter', '')

df = dpc.data_2_df(questions_dir, subject_dir)
df = dpc.preprocessing(df)
df = dpc.link_outcome_labels(df,
                              label_file='importance_labels.csv',
                              data_directory=ARTICLE_DIR + os.sep)

# ---------------------------------------------------------------------------
# Merge kpthesaurus from COMMUNICATEDCASES overlap metadata
# ---------------------------------------------------------------------------
comm_meta_path = os.path.join(ARTICLE_DIR, 'overlap_cases', 'pruned_COMMUNICATEDCASES_meta.json')
if os.path.exists(comm_meta_path):
    meta = pd.read_json(comm_meta_path, lines=True)[['itemid', 'kpthesaurus', 'kpdate']]
    meta.rename(columns={'itemid': 'Filename', 'kpdate': 'doc_date'}, inplace=True)
    df = pd.merge(df, meta, on='Filename', how='left')
else:
    print(f'Warning: {comm_meta_path} not found — kpthesaurus unavailable, skipping keyword filter.')
    df['kpthesaurus'] = ''
    df['doc_date'] = pd.NaT

# ---------------------------------------------------------------------------
# Filter to article-specific keywords
# ---------------------------------------------------------------------------
article_keywords = cfg.KEYWORDS

df['keyword_num'] = df['kpthesaurus'].fillna('').apply(
    lambda x: [k.strip() for k in x.split(';') if k.strip()])

df_article = df[df['keyword_num'].apply(
    lambda ks: any(k in article_keywords for k in ks))].copy()

# Plain-text keyword columns
df_article[f'keywords_art_{cfg.ARTICLE_NUM}'] = df_article['keyword_num'].apply(
    lambda ks: [k for k in ks if k in article_keywords])

df_article[f'keywords_art_{cfg.ARTICLE_NUM}_text'] = df_article[f'keywords_art_{cfg.ARTICLE_NUM}'].apply(
    lambda ks: ', '.join(label_to_keyword.get(k, k) for k in ks))

# Minimum subject-matter word count
df_article['Subj_Count'] = df_article['Subject Matter'].str.split().str.len()
df_article = df_article[df_article['Subj_Count'] >= 50]

df_article = df_article.drop(columns=['keyword_num'])

print(f'Cases after filtering: {len(df_article)}')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(ARTICLE_DIR, 'comm_cases.pkl')
df_article.to_pickle(out_path)
print(f'Saved comm_cases.pkl → {out_path}')

# Validation / test split (80/20)
valid = df_article.sample(frac=0.2, random_state=42)
test  = df_article.drop(valid.index)
valid.to_pickle(os.path.join(ARTICLE_DIR, 'comm_cases_valid.pkl'))
test.to_pickle(os.path.join(ARTICLE_DIR, 'comm_cases_test.pkl'))
print(f'  Valid: {len(valid)}  Test: {len(test)}')
