#TO DO - Add metadata

"""
Version history
v1_0 = processes communicated case data for Article 3
"""

import argparse
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as dpc
import json

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data/')
ap.add_argument('--keywords', default='350,89,90,596,620,618,192,193,633,492')
ap.add_argument('--output', default=None)
args = ap.parse_args()

#set directory
data_directory = args.data_dir

# Set the directory for the txt files
questions = data_directory +'corpora/communication_phase/questions/'
subject_matter = data_directory + 'corpora/communication_phase/subject_matter/'

#convert txt files to df
df = dpc.data_2_df(questions, subject_matter)

#preprocess df
df = dpc.preprocessing(df)

#link to importance labels and date
df = dpc.link_outcome_labels(df,label_file='important_labels.csv',data_directory=data_directory)

#article keywords
article3 = args.keywords.split(',')

# Load the .json file for the keyword labels
with open(os.path.join(data_directory, 'key_labels.json')) as f:
    json_data = json.load(f)

# Create a dictionary to map keyword labels to keywords
label_to_keyword = {label: keyword for label,keyword in json_data.items()}

# Convert the key_words_keys column to a string
df['key_words_keys'] = df['key_words_keys'].astype(str)

# Create a new column to store the keywords in plain text
_keywords_col = []
for i in range(len(df)):
    keys = str(df['key_words_keys'].iloc[i]).split(';')
    keywords = []
    for key in keys:
        if key in label_to_keyword.keys():
            keywords.append(label_to_keyword[key])
        else:
            pass
    _keywords_col.append(keywords)
df['keywords'] = _keywords_col

# Filter out the rows with no keywords
df_2 = df[df['keywords'].apply(lambda x: len(x) > 0)]

# Create a new column to store keyword nums seperated in a list
df_2['keyword_num'] = df_2.apply(lambda x: x['key_words_keys'].split(';'), axis=1)

# Filter out the rows with no keywords in Article 3
df_art_3 = df_2[df_2['keyword_num'].apply(lambda x: any(x in article3 for x in x))]

# Create a new column to store the keywords relevant to Article 3
df_art_3['keywords_art_3'] = df_art_3['keyword_num'].apply(lambda x: [i for i in x if i in article3])

# Create a new column to store the keywords relevant to Article 3 in plain text
df_art_3['keywords_art_3_text'] = df_art_3['keywords_art_3'].apply(lambda x: ', '.join([label_to_keyword[key] for key in x if key in label_to_keyword.keys()]))

# Filter out some columns
df_art_3 = df_art_3[['Filename', 'Questions', 'Subject Matter', 'appno','source_file','doc_date','importance','keywords_art_3','keywords_art_3_text']]

# Create a new column to store the word count of the subject matter
df_art_3['Subj_Count'] = df_art_3['Subject Matter'].str.split().str.len()

#ensure word count >= 50 for subject matter
df_art_3 = df_art_3[df_art_3['Subj_Count'] >= 50]

#save comm cases
_default_out = os.path.join(data_directory, 'comm_cases.pkl')
df_art_3.to_pickle(args.output if args.output else _default_out)