import pandas as pd 
from openai import OpenAI
import pandas as pd
import os
import json
from sklearn import model_selection
import sys
sys.path.insert(0,'/users/sgdbareh/volatile/ECHR_Importance')
from GPT_Experiments import Experiment_2, save_file, generate_example_candidates, create_examples

JSON_SCHEMAS = [{"Case Importance":"string (select one of: key_case, 1, 2, 3)", "Reasoning":"string (give your reason for the importance)" },
                {"Case Importance Under Consideration": "string (select one of: key_case, 1, 2, 3)","Confidence in Importance Ascription": "int 1-100", "Reasoning": "string (give your reason for or against the importance)" }
                ]

PARAMETERS = {'schema':[JSON_SCHEMAS[1]],
              'zero_shot':[True],
              'text':[1],
              'reasoning':[False],
              'random_state':[42]
              }

MAX_TOKENS = 1500
TEMPERATURE = 0
TOP_P = 1.0

data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')

#set param grid search - CHANGE
grid = model_selection.ParameterGrid(PARAMETERS)

for params in grid:

    match params['text']:
        case 1:
            content = 'subject_matter'
        case 2:
            content = 'questions'
        case 3:
            content = 'both'

    output = []

    for file in range(len(data)):

        filename = data.iloc[file]['Filename']


        if params['schema'] == JSON_SCHEMAS[0]:
            exp_2 = Experiment_2(data,content=content,grand_chamber=False,reasoning=params['reasoning'])
            output = exp_2.run_async(schema=params['schema'],zero_shot=params['zero_shot'],text=params['text'],info=True,temperature=0,max_tokens=1500)

        else:
            row = data.iloc[file]
            text = row['Subject Matter']
            prompt = f'''You are a lawyer in the European Court of Human Rights, and your goal is to predict the importance of a case, based on information provided from a communicated case. Importance in a legal setting refers to the significance of a case in terms of its impact on the development of case law. All the cases concern Article 3 of the European Convention of Human Rights, about the prohibition of torture.
                        The following information is provided to you:
                        You will be given a communicated case, including the subject matter of the case.
                        You are given a description of the different levels of importance: key_case: These are the most important and have been selected as key cases and have been selected for publication in the Court\'s official reports; 
                                            1: The case is of high importance. The case makes a significant contribution to the development, clarification or modification of its case law, either generally or in relation to a particular case; 
                                            2: The case is of medium importance. The case while not making a significant contribution to the case-law, nevertheless it goes beyond merely applying existing case law; 
                                            3: The case is of low importance. The case is of limited interest and simply applies existing case law.
                        Based on the information given to you predict the importance of the case according to the criteria given. 
                        If you do not know the importance, state that you do not have enough information.
                        The output should be given directly in JSON format, with the following schema:
                        {params['schema']},
                        The communicated case information you should base your judgement on is as follows: {text}
                        Give a separate output for each importance category: 3, 2, 1, key_case.'''

            #strip whitespace
            prompt = prompt.strip()

            template = {"custom_id": f'{filename}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": prompt}],'response_format':{'type': 'text'},'max_tokens':MAX_TOKENS,'temperature':TEMPERATURE,'top_p':TOP_P, 'seed':params['random_state']}}

            output.append(template)   
        
    
    #save batch file
    save_file(output,filepath='/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/BASELINE_ITER',batch_name='TEST_prediction_iterative',experiment = 2, zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'],identifier='random_state_{}'.format(params['random_state']))
    