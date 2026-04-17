##BACKUP OF RAG CODE

import pandas as pd 
import os
import json
from sklearn import model_selection
import sys
sys.path.insert(0,'/users/sgdbareh/volatile/ECHR_Importance')
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

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout

parser = OptionParser(usage='usage: -k K -e embedding -b BERT')   
parser.add_option("-k", "--K", action = "store", type = "int", dest = "k", default = 3)
parser.add_option("-e", "--embedding_name", action = "store", type = "string", dest = "embedding_name",default = 'OPENAI-512-MMR')
parser.add_option("-t", action="store_true", dest="bert", default=True)
parser.add_option("-f", action="store_false", dest="bert")
(options, _) = parser.parse_args()

MAX_TOKENS = 1000
TEMPERATURE = 0
TOP_P = 1.0
K = options.k
EMBEDDING = options.embedding_name
BERT = options.bert

JSON_SCHEMAS = [{"Case Importance":"string (select one of: key_case, 1, 2, 3)", "Reasoning":"string (give your reason for the importance)" }
                ]

PARAMETERS = {'schema':[JSON_SCHEMAS[0]],
              'zero_shot':[False],
              'text':[1],
              'reasoning':[False],
              'random_state':[42]
              }

#RUN TEST VECTOR DATABASES
def select_embedding(embedding):
    if embedding == 'OPENAI-512-MMR':
        return pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/MMR_openai_raw_chunk_512_results.pkl')
    elif embedding == 'LBERT-512-MMR':
        return pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/MMR_legal-bert_raw_chunk_512_results.pkl')
    elif embedding == 'OPENAI-2048-COSINE':
        return pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/cosine_openai_raw_chunk_2048_results.pkl')
    elif embedding == 'BM25':
        return pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/BM25_results.pkl')
    else:
        raise ValueError('Invalid embedding')

def similar_cases_process():
    similar_cases = []

    for i in embedding:
        if i == '':
            continue  

        similar_cases.append(i.split(';')[0]) 

    return similar_cases 

def find_file_no(similar_cases, outcome_cases):

    file_numbers = {}
            
    #find fileNo and importance for most recent appNo case
    for i in similar_cases:
        cases = outcome_cases[outcome_cases['appno'].str.contains(i)]
        cases.sort_values(by='date',ascending=True,inplace=True)

        importance = cases.iloc[-1]['importance']
        fileNo = cases.iloc[-1]['File']

        file_numbers[fileNo] = importance

    return file_numbers

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

def relevance_check(test_case,example_candidates,fileNos):

    predict_BERT = []

    for i in example_candidates:

        predict_BERT.append(InputExample(texts=[str(test_case).lower(),str(i).lower()]))
    
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(predict_BERT, name='Relevance BERT')
    scores = [int(score > 0.5) for score in model.predict(evaluator.sentence_pairs)]

    #if scores.count(1) >= K:
        #logger.info(f'File: {filename} Relevance: {scores.count(1)}')
    fileNos = {list(fileNos.keys())[i]: list(fileNos.values())[i] for i, score in enumerate(scores) if score == 1}
    return dict(list(fileNos.items())[:K])
    
    #else:
    #    fileNos = {list(fileNos.keys())[i]: list(fileNos.values())[i] for i, score in enumerate(scores) if score == 1}
    #    return dict(list(fileNos.items())[:K])
    
# Load the data
test_data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')
test_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/comm_test_summaries.pkl')

assert len(test_data) == len(test_summaries)

# Load the outcome cases
outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')
outcome_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/outcome_summaries.pkl')

assert len(outcome_cases) == len(outcome_summaries)

#set param grid search
grid = model_selection.ParameterGrid(PARAMETERS)

embedding_location = select_embedding(EMBEDDING)

model = CrossEncoder(f"/users/sgdbareh/volatile/ECHR_Importance/BERT-rerank/model2024-11-07_14-39-42_FINAL")       

for params in grid:

    final_examples = {}

    match params['text']:
        case 1:
            content = 'subject_matter'
        case 2:
            content = 'questions'
        case 3:
            content = 'both'

    exp_2 = Experiment_2(test_data,content=content,grand_chamber=False,reasoning=params['reasoning'])

    output = []

    for file in range(len(test_data)):

        #get the row
        filename = test_data.iloc[file]['Filename']

        #get the similar cases
        embedding = embedding_location[filename]

        #relevance vs w/o relevance
        if BERT:

            fileNos = []

            K_try = K
            
            while len(fileNos) != K:

                K_try = K_try * 3

                similar_cases = similar_cases_process()
                similar_cases = similar_cases[:K_try]

                logging.info(f'File: {filename} Similar Cases: {len(similar_cases)}')

                file_numbers = find_file_no(similar_cases, outcome_cases)
                
                #generate example candidates
                example_candidates, _, fileNos = example_gen(outcome_summaries, file_numbers,word_count=200)
                
                #find test case summary text 200
                test_case = test_summaries[test_summaries['Filename'] == filename].iloc[0]['200 Word Summary']

                fileNos = relevance_check(test_case,example_candidates,file_numbers)

                if len(fileNos) > K:
                    raise ValueError('Too many examples')          
                
                if K_try > 100:
                    logging.info(f'Too many iterations - not enough relevant cases {len(fileNos)}')
                    break


            #THEN FIND 500 WORD FINAL EXAMPLES CORRESPONDING TO K
            examples, importance, _ = example_gen(outcome_summaries, fileNos,word_count=500)

        else:

            K_try = K
            examples = []

            while len(examples) != K:

                similar_cases = similar_cases_process()
                similar_cases = similar_cases[:K_try]

                logging.info(f'File: {filename} Similar Cases: {len(similar_cases)}')

                file_numbers = find_file_no(similar_cases, outcome_cases)

                examples, importance, _ = example_gen(outcome_summaries, file_numbers)


                K_try += 1

                if K_try > K*2:
                    raise ValueError('Too many iterations - not enough relevant cases')

            #assert len(examples) == K
        
        logging.info(f'File: {filename} Examples: {len(examples)}')
        examples_importance_dict = {examples[i]: importance[i] for i in range(len(examples))}

        final_examples[filename] = examples_importance_dict

        prompt = exp_2.get_rag_prompt(test_data.iloc[file], params['schema'], zero_shot=params['zero_shot'], text=params['text'], examples=examples_importance_dict)

        #strip whitespace
        prompt = prompt.strip()

        template = {"custom_id": f'{filename}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':MAX_TOKENS,'temperature':TEMPERATURE,'top_p':TOP_P, 'seed':params['random_state']}}

        output.append(template)       
        
    if BERT:
        folder = 'TEST_SEMANTIC_RELEVANCE'
    else:   
        folder = 'TEST_SEMANTIC_NO_RELEVANCE'

    with open(f'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/{folder}/final_examples_{K}_{EMBEDDING}_EXAMPLE_DICT.pkl', 'wb') as f:
        pickle.dump(final_examples, f)

    #save batch file
    #save_file(output,filepath=f'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/{folder}',batch_name=f'predictions_{K}_{EMBEDDING}',experiment = 'pred', zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'])































