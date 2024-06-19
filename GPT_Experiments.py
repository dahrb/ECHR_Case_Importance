"""
To Do:
- explore different roles
- explore different prompts both few-shot and zero-shot
- explore different temperature and max_tokens

Version history:
v1_0 = Initial version - Experiment 1 - bias detection setup and running both for sync and async use-cases
"""

from API_key import openai_key
from openai import OpenAI
import pandas as pd
import os
import json

JSON_SCHEMAS = [{"Case Importance":"int (1-4)","Summary":"string (description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Case Importance":"int (1-4)"},
                {"Case Importance":"string (select one of: key_case, 1, 2, 3)"}
                ]

class Experiment_1():

    '''
    Experiment 1 - detecting existing bias concerning cases in the dataset
    Use either in run_sync or run_async mode depending on whether you want instant results or to use the batch API
    We primarily use the batch API for this experiment
    '''

    def __init__(self,data):
        self.data = data
        self.process_data()

    def run_sync(self,schema,prompt_num,temperature=0,max_tokens=500):
        '''
        sends the experiment to the API for synch results
        '''
        
        counter = 0

        output = pd.DataFrame()

        for file in range(len(self.exp1_data)):
            prompt = self.get_prompt(schema, self.exp1_data['docname'][file], self.exp1_data['appno'][file], prompt_type=prompt_num)

            response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        response_format={'type': 'json_object'},
                        max_tokens=max_tokens,
                        temperature=temperature
                        )
            
            content = response.choices[0].message.content
            
            try:
                data = json.loads(content)
                temp_df = pd.DataFrame(data,index=[0])
                output = pd.concat([output,temp_df],ignore_index=True)

            except (json.JSONDecodeError, IndexError):
                raise IndexError(f'Error in decoding JSON response: {data}')          

            counter += 1

            if counter == 2:
                return output

    def run_async(self,batch_name,schema,prompt_num,temperature=0,max_tokens=500):
        '''
        formats the experiment to the batch API for synch results - still requires the file to be sent to the batch API
        '''
        
        output = []

        for file in range(len(self.exp1_data)):
            prompt = self.get_prompt(schema, self.exp1_data['docname'][file], self.exp1_data['appno'][file], prompt_type=prompt_num)

            template = {"custom_id": f'{self.exp1_data.index[file]}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':max_tokens,'temperature':temperature}}

            output.append(template)

        with open(f'./Batches/experiment_1_valid/{batch_name}_{prompt_num}.jsonl', 'w') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')

    def process_data(self):
        metadata = pd.read_json('./Data/overlap_cases/pruned_COMMUNICATEDCASES_meta.json',lines=True)
        metadata.rename(columns={'itemid':'Filename'}, inplace=True)
        self.exp1_data = pd.merge(self.data, metadata, on='Filename')
        self.exp1_data = self.exp1_data[['Filename','importance_x','appno','docname']]
        self.exp1_data.rename(columns={'importance_x':'importance'}, inplace=True)

    def get_prompt(self, schema, name, appnos, prompt_type='first'):

        match prompt_type:
            #option to say don't know, no info on HUDOC given
            case 'first':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. 
                    Using the information given to you tell me the case importance giving a response of either: key_case, 1, 2, 3. 
                    If you do not know the importance, state that you do not have enough information.
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #option to say don't know, info on HUDOC given
            case 'second':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. The case importance is part of the metadata on HUDOC.
                    Using the information given to you tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    If you do not know the importance, state that you do not have enough information.
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #no option to say don't know, info on HUDOC given
            case 'third':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. The case importance is part of the metadata on HUDOC.
                    Using the information given to you tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #no option to say don't know, no info on HUDOC given
            case 'fourth':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights. These values are given to every case after a judgment
                    or decision has been received. 
                    Using the information given to you tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''
            #demands LLM to use HUDOC information, no option to say don't know
            case 'fifth':
                exp1_prompt = f''' You are going to tell me the importance of the cases in the European Court of Human Rights.
                    Find the information from HUDOC and tell me the importance of the case giving a response of either: key_case, 1, 2, 3. 
                    The output should be given directly in JSON format, with the following schema: {schema}.
                    The name of the case is {name} and the application number/s is/are {appnos}.
                    '''

        return exp1_prompt



if __name__ == "__main__":
    
    # ASYNC EXAMPLE - EXPERIMENT 1
    client = OpenAI(api_key=openai_key)
    df = pd.read_pickle('valid_data.pkl')
    df_main = df[:-4]
    exp_1 = Experiment_1(df_main)
    
    for num in ['first','second','third','fourth','fifth']:
        exp_1.run_async(batch_name='experiment_1',schema=JSON_SCHEMAS[2],prompt_num=num,temperature=0,max_tokens=500)

    #submit files to BATCH API
    
    # SYNC EXAMPLE
    # output = exp_1.run_sync(JSON_SCHEMAS[0],'first',temperature=0,max_tokens=500)
    # output.to_pickle('exp_1_output.pkl')
    # print(output)


