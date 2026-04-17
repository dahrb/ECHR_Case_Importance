import pandas as pd 
import os
import json
from sklearn import model_selection
import sys
import torch
sys.path.insert(0,'/users/sgdbareh/volatile/ECHR_Importance')
sys.path.insert(0,'/users/sgdbareh/volatile/ECHR_Importance/Knowledge_Graph')

from GPT_Experiments import Experiment_2, save_file, generate_example_candidates, create_examples
from optparse import OptionParser
import pickle
os.chdir('/users/sgdbareh/volatile/ECHR_Importance')
from API_key import openai_key
from openai import OpenAI
client = OpenAI(api_key=openai_key)
import faiss
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
from network_TGN import Model
import time
import random
random.seed(42)

parser = OptionParser(usage='usage: -k K -e embedding -b BERT')   
parser.add_option("-k", "--K", action = "store", type = "int", dest = "k", default = 3)

(options, _) = parser.parse_args()
MAX_TOKENS = 1000
TEMPERATURE = 0
TOP_P = 1.0
K = options.k

JSON_SCHEMAS = [{"Case Importance":"string (select one of: key_case, 1, 2, 3)", "Reasoning":"string (give your reason for the importance)" }
                ]

PARAMETERS = {'schema':[JSON_SCHEMAS[0]],
              'zero_shot':[False],
              'text':[1],
              'reasoning':[False],
              'random_state':[42]
              }

def example_gen(outcome_summaries, file_numbers, word_count = 500):

    examples = []

    #find the case summary text
    summaries = outcome_summaries[outcome_summaries['Filename'].isin(file_numbers.keys())]

    for i in summaries.iterrows():
        examples.append(i[1][f'{word_count} Word Summary'])
    
    importance = list(file_numbers.values())
    fileNos = list(file_numbers.keys())

    assert len(examples) == len(importance) == len(fileNos)

    return examples, importance, fileNos

# Load the data
test_data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')
test_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/comm_test_summaries.pkl')

assert len(test_data) == len(test_summaries)

# Load the outcome cases
outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')
outcome_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/outcome_summaries.pkl')

assert len(outcome_cases) == len(outcome_summaries)

# Read in the pickle file
with open('/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/results_gold.pkl', 'rb') as f:
    gold_standard = pickle.load(f)

final_examples = {}

exp_2 = Experiment_2(test_data,content='subject_matter',grand_chamber=False,reasoning=PARAMETERS['reasoning'][0])
output = []

for file in range(len(test_data)):
    
    #get the row
    filename = test_data.iloc[file]['Filename']
    
    if filename in gold_standard:
        file_numbers = gold_standard[filename]
        #file_numbers = dict(list(file_numbers.items())[:K])
        if len(file_numbers) > K:
            file_numbers = dict(random.sample(list(file_numbers.items()), K))
        
        examples, importance, fileNos = example_gen(outcome_summaries, file_numbers)
        #print(f"File Numbers: {fileNos}")
    else:
        print(f"Filename {filename} not found in gold_standard.")
        
    examples_importance_dict = {examples[i]: importance[i] for i in range(len(examples))}

    final_examples[filename] = examples_importance_dict
    
    #print(examples_importance_dict)

    prompt = exp_2.get_rag_prompt(test_data.iloc[file], PARAMETERS['schema'][0], zero_shot=PARAMETERS['zero_shot'][0], examples=examples_importance_dict)

    #strip whitespace
    prompt = prompt.strip()

    template = {"custom_id": f'{filename}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':MAX_TOKENS,'temperature':TEMPERATURE,'top_p':TOP_P, 'seed':PARAMETERS['random_state'][0]}}

    output.append(template)
    
folder = 'GOLD_STANDARD'

with open(f'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/{folder}/final_examples_{K}_EXAMPLE_DICT.pkl', 'wb') as f:
    pickle.dump(final_examples, f)

#save batch file
save_file(output,filepath=f'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/{folder}',batch_name=f'predictions_{K}',experiment = 'pred', zero_shot=PARAMETERS['zero_shot'][0],text=PARAMETERS['text'][0],schema=PARAMETERS['schema'][0])


