"""
Version history
v1_1 = implements ability to sample specific cases to correspond with colleague's summarisation task
v1_0 = implements following features: load data; create dataframe; preprocessing; prepare data for experiments; sample data for validation and test sets
"""

import pandas as pd
import numpy as np
import os

def sampleHelper(df, importance, size):

    return df[df['importance'] == importance].sample(size, random_state=154)

def sample(df:pd.DataFrame, sampling_dist:list = [4,4,10,32], appNos = []) -> pd.DataFrame:

    '''
    Function to sample the data based on the importance of the case

    Parameters:
    df: pd.DataFrame - the dataframe to sample from
    sampling_dist: list - the distribution of the sampling for each importance level
    appNos: list - the list of application numbers to sample if we want to include specific cases in our validation set
    '''

    sample_list = []

    for imp in df['importance'].unique():
        if imp == 1:
            sample_list.append(sampleHelper(df, imp, sampling_dist[0]))
        elif imp == 2:
            sample_list.append(sampleHelper(df, imp, sampling_dist[1]))
        elif imp == 3:
            sample_list.append(sampleHelper(df, imp, sampling_dist[2]))
        else:
            sample_list.append(sampleHelper(df, imp, sampling_dist[3]))

    if len(appNos) > 0:
        sample_list.append(df[df['appno'].isin(appNos)])

    # Check for duplicates in sample_list
    df = pd.concat(sample_list)
    if df.duplicated().any():
        raise ValueError("Duplicates found in sample_list.")

    return df

def data_2_df(questions_dir, subject_matter_dir):
    
    # Initialize an empty list to store the data
    questions_list = []
    subject_matter_list = []

    folders = [questions_dir, subject_matter_dir]

    for folder in folders:
        if not os.path.exists(folder):
            print(f"Error: {folder} does not exist")
            return None
        
        # Iterate over the txt files
        for file in os.listdir(folder):
            # Read the txt file into a dataframe
            with open(folder + file, 'r') as text:
                data = [file, text.read()]
            # Append the data to the respective lists
            if folder == questions_dir:
                questions_list.append(data)
            else:
                subject_matter_list.append(data)

    #questions_list
    questions_df = pd.DataFrame(questions_list,columns=['Filename','Questions'])
    subject_matter_df = pd.DataFrame(subject_matter_list,columns=['Filename','Subject Matter'])

    subject_matter_df['Filename'] = subject_matter_df['Filename'].str.replace('.txt', '')
    questions_df['Filename'] = questions_df['Filename'].str.replace('.txt', '')

    # create merged df
    df = pd.merge(questions_df, subject_matter_df, on='Filename')
    df = df.rename(columns={'Word Count_x': 'Question_Count', 'Word Count_y': 'Subject_Matter_Count'})

    return df

def preprocessing(df:pd.DataFrame):

    #remove whitespace
    df['Questions'] = df['Questions'].str.strip()
    df['Subject Matter'] = df['Subject Matter'].str.strip()

    #remove repeat phrases
    df['Subject Matter'] = df['Subject Matter'].str.replace('THE FACTS\n', '')
    df['Subject Matter'] = df['Subject Matter'].str.replace('\n', '')
    df['Questions'] = df['Questions'].str.replace('\n', '')

    return df

def link_outcome_labels(df:pd.DataFrame,label_file = 'importance_labels.csv',data_directory = './Data/'):

    #link outcome label
    labels = pd.read_csv(data_directory + label_file)
    labels = labels.rename(columns={'itemid':'Filename'})

    df = pd.merge(df, labels, on='Filename')

    return df

if __name__ == '__main__':

    # Code to preprocess data for initial GPT-4o experiments
    # Sample 50 cases with specified dist + 4 specific example cases for few shot
    # Rest of data availiable as test data for prediction
    data_directory = './Data/'

    # Set the directory for the txt files
    questions = data_directory +'questions/'
    subject_matter = data_directory + 'subject_matter/'

    df = data_2_df(questions, subject_matter)

    df = preprocessing(df)

    df = link_outcome_labels(df)

    sample_df = sample(df,sampling_dist=[4,4,10,32] ,appNos=['6697/18','35589/08','12427/22','19866/21'])

    valid_data = sample_df[['Filename','Questions','Subject Matter','importance']]

    test_data = df.drop(sample_df.index)
    test_data = test_data[['Filename','Questions','Subject Matter','importance']]

    valid_data.to_pickle('valid_data.pkl')
    test_data.to_pickle('test_data.pkl')