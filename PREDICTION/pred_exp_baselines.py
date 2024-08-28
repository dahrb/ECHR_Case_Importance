import pandas as pd 
from openai import OpenAI
import pandas as pd
import os
import json
from sklearn import model_selection
import sys
sys.path.insert(0,'/users/sgdbareh/volatile/ECHR_Importance')
from API_key import openai_key
from GPT_Experiments import Experiment_2, save_file, generate_example_candidates, create_examples
from optparse import OptionParser

JSON_SCHEMAS = [{"Case Importance":"string (select one of: key_case, 1, 2, 3)", "Reasoning":"string (give your reason for the importance)" }
                ]

PARAMETERS = {'schema':[JSON_SCHEMAS[0]],
              'zero_shot':[True,False],
              'text':[1,2,3],
              'reasoning':[True,False]
              }

#connect to api key
client = OpenAI(api_key=openai_key)

# Load the data
data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')

train_data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Vector_DB/test.pkl')

#set param grid search
grid = model_selection.ParameterGrid(PARAMETERS)

for params in grid:

    match params['text']:
        case 1:
            content = 'subject_matter'
        case 2:
            content = 'questions'
        case 3:
            content = 'both'

    exp_2 = Experiment_2(data,content=content,grand_chamber=False,reasoning=params['reasoning'])
    #set examples
    example_candidates = generate_example_candidates(train_data,keyword='importance',labels=[1,2,3,4])
    examples = create_examples(train_data,text=params['text'],importance=True)
    #create batch files for given prompt + parameters
    output = exp_2.run_async(schema=params['schema'],zero_shot=params['zero_shot'],text=params['text'],examples=examples,info=True,temperature=0,max_tokens=1500)
    #save batch file
    save_file(output,filepath='/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches',batch_name='prediction_baselines',experiment = 2, zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'],identifier='{}_REASONING'.format(params['reasoning']))
