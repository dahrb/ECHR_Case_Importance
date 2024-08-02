"""
Version history
v1_0 = creates the outcome text data for Article 3
"""

#Column Guide -
#doctypebranch = admissibility etc
#respondent = country
#appno = application number
#item_id = unique id
#decisiondate = date of decision
#judgementdate = date of judgement
#extractedappno = citations (need to exclude actual appno/s of case)
#conclusion = outcome
#importance = importance
#kpthesaurus = keywords

import pandas as pd
import os
import sys
sys.path.insert(0,'/users/sgdbareh/volatile/ECHR_Importance')

#set directory
data_directory = '/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data/corpora/article3'

#article 3 keywords
article3 = ['350','89','90','596','620','618','192','193','633','492']

#add filenames to a list then create lists of the facts and law section seperately
list_files = []

for folder in os.listdir(data_directory):
    subfolder = data_directory + '/' + folder
    for file in os.listdir(subfolder):
        list_files.append(subfolder+'/'+file)

facts_list = []
law_list = []

for file in list_files:
    
    if file.split('/')[-1] == 'fact_section':
        for text_file in os.listdir(file):
            with open(file + '/' + text_file,'r') as text:
                data = [text_file, text.read()]
            facts_list.append(data)

    else:
        for text_file in os.listdir(file):
            with open(file + '/' + text_file,'r') as text:
                data = [text_file, text.read()]
            law_list.append(data)

#convert facts and law to dfs
facts_df = pd.DataFrame(facts_list,columns=['Filename','Facts'])
law_df = pd.DataFrame(law_list,columns=['Filename','The Law'])

#data preprocessing/cleaning
facts_df['Filename'] = facts_df['Filename'].str.replace('.txt', '')
law_df['Filename'] = law_df['Filename'].str.replace('.txt', '')
facts_df['Facts'] = facts_df['Facts'].str.strip()
law_df['The Law'] = law_df['The Law'].str.strip()
facts_df['Facts'] = facts_df['Facts'].str.replace('\n', '')
law_df['The Law'] = law_df['The Law'].str.replace('\n', '')

#word count of facts
facts_df['Word Count'] = facts_df['Facts'].str.split().str.len()

#merge facts and law dfs
df_merged = pd.merge(facts_df, law_df, on='Filename',how='inner')

#filter out cases with less than 60 words
df_merged = df_merged[df_merged['Word Count'] >=60]

#extract date and file name
df_merged['date'] = df_merged['Filename'].str.split('_').str[0]
df_merged['File'] = df_merged['Filename'].str.split('_').str[1]

#convert date to datetime
df_merged['date'] = pd.to_datetime(df_merged['date'], format='%Y-%m-%d')

#filter out cases before 1995
df_merged = df_merged[df_merged['date'] >= '1995-01-01']

#create columns for metadata .json
metadata = pd.DataFrame(columns=['itemid','appno','doctypebranch','respondent','decisiondate','extractedappno','conclusion','importance','kpthesaurus','judgementdate'])

#load metadata
for file in os.listdir('/users/sgdbareh/volatile/ECHR_Importance/raw_case_metadata'):
    if file.endswith('.json'):
        with open('/users/sgdbareh/volatile/ECHR_Importance/raw_case_metadata' + '/' + file) as f:
            data = pd.read_json(f,lines=True)
        metadata = pd.concat([metadata,data])

#reset index
metadata.reset_index(drop=True,inplace=True)

#filter out unneeded columns
filtered_metadata = metadata.iloc[:, :10]

#convert dates to datetime
filtered_metadata['judgementdate'] =  pd.to_datetime(filtered_metadata['judgementdate'],format='%d/%m/%Y %H:%M:%S').dt.date
filtered_metadata['decisiondate'] =  pd.to_datetime(filtered_metadata['decisiondate'],format='%d/%m/%Y %H:%M:%S').dt.date
filtered_metadata['decisiondate'] = pd.to_datetime(filtered_metadata['decisiondate'])
filtered_metadata['judgementdate'] = pd.to_datetime(filtered_metadata['judgementdate'])

#fill missing judgement dates with decision dates
filtered_metadata['judgementdate'] = filtered_metadata['judgementdate'].fillna(filtered_metadata['decisiondate'])

#drop decision date
filtered_metadata.drop(['decisiondate'],axis=1,inplace=True)

#rename itemid to File
filtered_metadata.rename(columns={'itemid':'File'},inplace=True)

#drop duplicate filenames
filtered_metadata.drop_duplicates(subset='File', keep=False, inplace=True)

#filter out metadata for cases before 1995
filtered_metadata = filtered_metadata[filtered_metadata['judgementdate'] >= '1995-01-01']

#merge two dataframes - metadata and text data
full_outcome_data = pd.merge(df_merged,filtered_metadata,on='File',how='inner')

#drop unneeded columns
full_outcome_data.drop(['Filename','judgementdate'],axis=1,inplace=True)

#filter out cases with no keywords
full_outcome_data = full_outcome_data[full_outcome_data['kpthesaurus'] != '']

#filter out cases with no Art. 3 keywords
full_outcome_data = full_outcome_data[full_outcome_data['kpthesaurus'].apply(lambda x: any(keyword in x for keyword in article3))]

#save to .pkl
full_outcome_data.to_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')




