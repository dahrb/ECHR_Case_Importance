from openai import OpenAI
import pandas as pd
from API_key import openai_key
import json
from sklearn import model_selection

JSON_SCHEMAS = [{"200 Word Summary":"(string describing the key facts of the case in 200 words max)","500 Word Summary":"(string describing the key facts of the case in 500 words max)" },
                ]

def save_file(output,filepath,batch_name):

    with open(f'{filepath}/{batch_name}.jsonl', 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')

def prep_prompt(data, max_tokens=1500, temperature=0, top_p=1, prompt_type = 'comm'):
    
    output = []

    for file in range(len(data)):

        if prompt_type == 'comm':
            prompt = comm_prompt_generation(data.iloc[file])
        else:
            prompt = outcome_prompt_generation(data.iloc[file])

        try:
            id = data.iloc[file]['Filename']
        except:
            id = data.iloc[file]['File']

        template = {"custom_id": f'{id}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':max_tokens,'temperature':temperature,'top_p':top_p, 'seed':148}}
        output.append(template)

    return output


def comm_prompt_generation(row,schema:dict = JSON_SCHEMAS[0]):

    '''Function to generate a prompt for the GPT-4o model.
    
    Parameters: 
    row: pd.Series
        A row from the dataframe containing the data.
    schema: dict
        A dictionary containing the schema for the JSON output.
    Returns:
    prompt: str
        The prompt to be used for the GPT-4o model.
    '''

    prompt = f''' 
    You are a lawyer in the European Court of Human Rights, and your goal is to summarise communicated cases.
    The following information is provided to you:
    You will be given the 'Subject Matter' section of a communicated case. Please provide two summaries of the facts of the case, one
    with a maximum word count of 200 words and another with a maximum word count of 500 words. The summaries should be concise and capture the key facts of the case.
    The case relates to Article 3 of the European Convention of Human Rights, concerning the prohibition of torture.
    The case text is as follows: {row['Subject Matter']}. 
    The output should be given directly in JSON format, with the following schema: {schema}.
    '''

    return prompt

def outcome_prompt_generation(row,schema:dict = JSON_SCHEMAS[0]):

    '''Function to generate a prompt for the GPT-4o model.
    
    Parameters: 
    row: pd.Series
        A row from the dataframe containing the data.
    schema: dict
        A dictionary containing the schema for the JSON output.
    Returns:
    prompt: str
        The prompt to be used for the GPT-4o model.
    '''

    prompt = f''' 
    You are a lawyer in the European Court of Human Rights, and your goal is to summarise outcome cases.
    The following information is provided to you:
    You will be given the 'Facts' and 'The Law' sections of a outcome case. Please provide two summaries of the facts of the case, one
    with a maximum word count of 200 words and another with a maximum word count of 500 words. The summaries should be concise and capture the key aspects of the case.
    The case relates to Article 3 of the European Convention of Human Rights, concerning the prohibition of torture.
    The 'Facts' section of the case is: {row['Facts']}.
    The 'The Law' section of the case is: {row['The Law']}.
    The output should be given directly in JSON format, with the following schema: {schema}.
    '''

    return prompt












