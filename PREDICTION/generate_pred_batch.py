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
parser.add_option("--KG", action="store_true", dest="KG",default=False)

(options, _) = parser.parse_args()

MAX_TOKENS = 1000
TEMPERATURE = 0
TOP_P = 1.0
K = options.k
EMBEDDING = options.embedding_name
BERT = options.bert
KG = options.KG

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

def find_importance_from_file_no(file_numbers):
    
    cases = outcome_cases[outcome_cases['File'].isin(file_numbers)]
    
    file_numbers = {}
    for i in cases.iterrows():
        file_numbers[i[1]['File']] = i[1]['importance']
    
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

def cosine(a,b):
    return torch.nn.functional.cosine_similarity(a, b)
            
  
def calculate_similarity(knowledge_graph, nodes, unique_value, node_feat=False):
    node_embeddings = [knowledge_graph.get_node_embedding_at_time(node, node_feat=node_feat) for node in nodes]
    unique_embedding = knowledge_graph.get_node_embedding_at_time(unique_value, node_feat=node_feat)
    similarities = [cosine(node_embedding, unique_embedding) for node_embedding in node_embeddings]
    return unique_value, sum(similarities)

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

#LOAD KNOWLEDGE GRAPH    
if KG:
    knowledge_graph = Model()
    knowledge_graph.load_data()
    checkpoint = torch.load('/users/sgdbareh/volatile/ECHR_Importance/Knowledge_Graph/best_model_False.pth')
    knowledge_graph.load_parameters(checkpoint)
    print('knowledge graph loaded')


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
            examples, importance, fileNos = example_gen(outcome_summaries, fileNos, word_count=500)

        else:

            K_try = K
            examples = []

            while len(examples) != K:

                similar_cases = similar_cases_process()
                similar_cases = similar_cases[:K_try]

                logging.info(f'File: {filename} Similar Cases: {len(similar_cases)}')

                file_numbers = find_file_no(similar_cases, outcome_cases)

                examples, importance, fileNos = example_gen(outcome_summaries, file_numbers)
                
                K_try += 1

                if K_try > K*2:
                    raise ValueError('Too many iterations - not enough relevant cases')

            #assert len(examples) == K
        
        #KNOWLEDGE GRAPH
        ############################################################################################################
        if KG:
            #converts filenames to integers
            converter = knowledge_graph.dataset.filename_to_int

            nodes = []
            
            for fileNo in fileNos:
                try:
                    nodes.append(converter[fileNo])
                except:
                    pass
            
            while len(nodes) < K:
                K_try = K_try * 2
                similar_cases = similar_cases_process()
                similar_cases = similar_cases[:K_try]

                logging.info(f'File: {filename} Similar Cases: {len(similar_cases)}')

                file_numbers = find_file_no(similar_cases, outcome_cases)

                examples, importance, fileNos = example_gen(outcome_summaries, file_numbers, word_count=500)
                #find test case summary text 200
                
                if BERT:
                    test_case = test_summaries[test_summaries['Filename'] == filename].iloc[0]['200 Word Summary']

                    fileNos = relevance_check(test_case,example_candidates,file_numbers)

                for fileNo in fileNos:
                    try:
                        nodes.append(converter[fileNo])
                    except:
                        pass

                if K_try > 100:
                    logging.info(f'Too many iterations - not enough relevant cases {len(fileNos)}')
                    break
            
            try:
                nodes = nodes[:K]
            except:
                pass
                
            #calc timestep of case relative to reference date
            reference_date = pd.Timestamp('1996-12-18')
            #print(test_data.columns)
            date = test_data.iloc[file]['doc_date']

            timestamp = (pd.Timestamp(date) - reference_date).days
            mask = knowledge_graph.data.t < timestamp
            filtered_data = knowledge_graph.data.edge_index[:, mask]
            # Extract unique values
            unique_values_tensor = torch.unique(filtered_data)

            # Convert the tensor to a list
            unique_values_list = unique_values_tensor.tolist()
            
            ###CREATE NEW SUB-GRAPH
            knowledge_graph.get_graph_at_time(timestamp)

            #29.14 seconds from 145 seconds - EXCELLENT

            # Calculate node similarity score
            node_similarity_score = {}

            # Sequential processing
            for unique_value in unique_values_list:
                unique_value, node_cum_sum = calculate_similarity(knowledge_graph, nodes, unique_value, False)
                node_similarity_score[unique_value] = node_cum_sum
                
            # Sort the node similarity score dictionary
            node_similarity_score = {k: v for k, v in sorted(node_similarity_score.items(), key=lambda item: item[1], reverse=True)}
            
            ########OLD CODE
            
            # ###CALCULATE NODE SIMILARITY SCORE
            # node_similarity_score = {}
   
            # for i in range(len(unique_values_list)):
            #     node_cum_sum = 0
            #     for j in range(len(nodes)):
            #         node_cum_sum += cosine(knowledge_graph.get_node_embedding_at_time(nodes[j], node_feat=False),
            #                                knowledge_graph.get_node_embedding_at_time(unique_values_list[i], node_feat=False))
            #     node_similarity_score[unique_values_list[i]] = node_cum_sum
            
            ########OLD CODE
           
            # node_similarity_score = {k: v for k, v in sorted(node_similarity_score.items(), key=lambda item: item[1], reverse=True)}
            # #print(node_similarity_score)

            most_similar_nodes = list(node_similarity_score.keys())
            most_similar_nodes = [node for node in most_similar_nodes if node not in nodes]
            most_similar_scores = [node_similarity_score[node] for node in most_similar_nodes]
            
            K_Nodes = most_similar_nodes[:K]
            
            # Convert the node indices back to filenames
            int_to_filename = {v: k for k, v in converter.items()}
            K_Filenames = [int_to_filename[node] for node in K_Nodes]

            print('Top K Filenames:', K_Filenames)
            print('Top K Scores:',most_similar_scores[:K])
            print('OG Nodes:',nodes)
        
            if BERT:
                #generate example candidates
                example_candidates, _, fileNos = example_gen(outcome_summaries, file_numbers,word_count=200)
                
                #find test case summary text 200
                test_case = test_summaries[test_summaries['Filename'] == filename].iloc[0]['200 Word Summary']

                fileNos = relevance_check(test_case,example_candidates,file_numbers)
                
                if len(fileNos) < K:
                    K_new = K
                    while len(fileNos) < K:
                        K_new = K_new*2
                        K_Nodes = most_similar_nodes[:K_new]
                        # Convert the node indices back to filenames
                        K_Filenames = [int_to_filename[node] for node in K_Nodes]
                        example_candidates, _, fileNos = example_gen(outcome_summaries, file_numbers,word_count=200)
                
                        #find test case summary text 200
                        test_case = test_summaries[test_summaries['Filename'] == filename].iloc[0]['200 Word Summary']

                        fileNos = relevance_check(test_case,example_candidates,file_numbers)
                        if K_new > 100:
                            logging.info(f'Too many iterations - not enough relevant cases {len(fileNos)}')
                            break
                        
            print(K_Filenames)
            
            file_numbers = find_importance_from_file_no(K_Filenames)

            examples, importance, fileNos = example_gen(outcome_summaries, file_numbers, word_count=500)
            
        #logging.info(f'File: {filename} Examples: {len(examples)}')
        examples_importance_dict = {examples[i]: importance[i] for i in range(len(examples))}

        final_examples[filename] = examples_importance_dict

        prompt = exp_2.get_rag_prompt(test_data.iloc[file], params['schema'], zero_shot=params['zero_shot'], text=params['text'], examples=examples_importance_dict)

        #strip whitespace
        prompt = prompt.strip()

        template = {"custom_id": f'{filename}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':MAX_TOKENS,'temperature':TEMPERATURE,'top_p':TOP_P, 'seed':params['random_state']}}

        output.append(template)  
        
    if BERT:
        folder = 'TEST_SEMANTIC_RELEVANCE'
        if KG:
            folder = 'KG_SEM_TEST'
    else:   
        folder = 'TEST_SEMANTIC_NO_RELEVANCE'
        if KG:
            folder = 'KG_NO_SEM_TEST'

    with open(f'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/{folder}/final_examples_{K}_{EMBEDDING}_EXAMPLE_DICT.pkl', 'wb') as f:
        pickle.dump(final_examples, f)

    #save batch file
    save_file(output,filepath=f'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/{folder}',batch_name=f'predictions_{K}_{EMBEDDING}',experiment = 'pred', zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'])































