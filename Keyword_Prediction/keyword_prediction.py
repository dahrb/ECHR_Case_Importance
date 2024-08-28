
from API_key import openai_key
from openai import OpenAI
import pandas as pd
import os
import json
from sklearn import model_selection

JSON_SCHEMAS = [{"Keywords":"list"},
                {"Keywords":"list","Reasons":"string"},]

PARAMETERS = {'schema':[JSON_SCHEMAS[0],JSON_SCHEMAS[1]],
              'zero_shot':[True,False],
              'text':[1,2,3]}


class Keyword_Prediction():

    '''
    '''

    def __init__(self,data):
        self.data = data

    def run_async(self, schema:dict = JSON_SCHEMAS[0], zero_shot:bool =True, examples:list = [], text = 3, info:bool=True, prompt_type:str='first',temperature:int=0,max_tokens:int=550,top_p=1):
        '''
        formats the experiment to the batch API for synch results - still requires the file to be sent to the batch API

        Parameters:
        filepath: str
            path to save the file
        batch_name: str
            name of the batch file
        experiment: int 
            experiment number can be 1 or 2
        schema: dict
            schema to be used for the JSON file
        zero_shot: bool
            determines if the experiment is zero-shot or few-shot
        text: int
            determines the text to be used in the prompt 1= Subject Matter, 2= Questions, 3= Both
        examples: list
            examples to be used in the prompt
        info: bool
            determines if the prompt includes the option to say don't know
        prompt_type: str
            determines the type of prompt to be used
        temperature: int
            determines the temperature to be used in the model
        max_tokens: int
            determines the max tokens to be used in the model
        '''
        
        output = []

        for file in range(len(self.data)):
    
            prompt = self.get_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            id = self.data.iloc[file]['Filename']
            template = {"custom_id": f'{id}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':max_tokens,'temperature':temperature,'top_p':top_p, 'seed':42}}
            output.append(template)
        
        return output

    def get_prompt(self, row,schema:dict = JSON_SCHEMAS[0], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):

        '''Function to generate a prompt for the GPT-4o model.
        '''

        match text:
            case 1:
                text = row['Subject Matter']
                text_amount = 'subject matter of the case'
            case 2:
                text = row['Questions']
                text_amount = 'questions asked to the parties'
            case 3:
                text = row['Subject Matter'] + ' ' + row['Questions']
                text_amount = 'subject matter of the case and the questions asked to the parties'
            case _:
                raise ValueError('Invalid text value. Please enter a value between 1 and 3.')

        if zero_shot:
            additional_context = ''
        else:
            additional_context = f'''You are also given a number of examples of some communicated cases and the keywords assigned to them.'''
            for i,j in examples.items():
                additional_context += f''' Case Example: {i}; Corresponding Keywords: {j}; '''
    
        keyword_list = ''' '350': (Art. 3) Prohibition of torture,
                        '89': (Art. 3) Degrading punishment,
                        '90': (Art. 3) Degrading treatment,
                        '596': (Art. 3) Effective investigation,
                        '620': (Art. 3) Expulsion,
                        '618': (Art. 3) Extradition,
                        '192': (Art. 3) Inhuman punishment,
                        '193': (Art. 3) Inhuman treatment,
                        '633': (Art. 3) Positive obligations,
                        '492': (Art. 3) Torture'''
        
        if info:
            state_info = 'If you do not know the importance, state that you do not have enough information.'
        else:
            state_info = ''

        prompt = f''' 
        You are a lawyer in the European Court of Human Rights, and you have been asked to assign keywords to different communicated cases based on which parts of Article 3 of the European Convention of Huyman Rights they may concern.
        Only assign keywords from the list you are given in their numerical form. You may use as many, or as few keywords as you believe are applicable for each case. All Article 3 cases have '350' as a keyword.
        The following information is provided to you:
        You will be given a communicated case, including the {text_amount}.
        You are given a description of the different keywords: {keyword_list}.
        {additional_context}.
        Based on the information given to you, please assign the keywords to the communicated case given. 
        {state_info}
        The output should be given directly in JSON format, with the following schema: {schema}.
        The communicated case information you should base your judgement on is as follows: {text}.
        '''

        return prompt
    
def create_examples(df,text=3):
    '''
    Function to create examples for the few-shot learning experiments
    '''
    examples = {}
    for i in range(len(df)):
        if text == 1:
            text = df.iloc[i]['Subject Matter']

        elif text == 2:
            text = df.iloc[i]['Questions']
        else:
            text = df.iloc[i]['Subject Matter'] + ' ' + df.iloc[i]['Questions']
            
        keywords = df.iloc[i]['keywords_art_3']

        examples[text] = keywords

    return examples

def save_file(output,filepath,batch_name, zero_shot=False,text=3,schema=JSON_SCHEMAS[0], identifier=None):
    if identifier is not None:
        identifier = identifier
    else:
        identifier = ''


    with open(f'{filepath}/{batch_name}_{zero_shot}_{text}_{JSON_SCHEMAS.index(schema)}_{identifier}.jsonl', 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    
    #connect to api key
    client = OpenAI(api_key=openai_key)
    #read in validation data
    df = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/comm_cases_valid.pkl')    
    #cuut off the last 4 cases as they are the ones used for few-shot examples in this setup
    df_example = df.sample(3,random_state=47)
    df_main = df.drop(df_example.index)

    #set params for experiment being run
    experiment = Keyword_Prediction(df_main)

    #set param grid search
    grid = model_selection.ParameterGrid(PARAMETERS)

    #set examples
    examples = create_examples(df_example,text=3)

    for params in grid:
        #create batch files for given prompt + parameters
        output = experiment.run_async( schema=params['schema'],zero_shot=params['zero_shot'],text=params['text'],examples=examples,info=True,temperature=0,max_tokens=550,top_p=1)
        #save batch file
        save_file(output,filepath='/users/sgdbareh/volatile/ECHR_Importance/Batches/keyword_prediction_2',batch_name='keyword_2', zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'])

    