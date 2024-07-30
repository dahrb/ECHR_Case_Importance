"""
Version history:
v1_3 = Implemented CoT experiments
v1_2 = Implemented variations on Experiment 2
v1_1 = Experiment 2 implementation
v1_0 = Initial version - Experiment 1 - bias detection setup and running for async use-cases
"""

from API_key import openai_key
from openai import OpenAI
import pandas as pd
import os
import json
from sklearn import model_selection


JSON_SCHEMAS = [{"Case Importance":"int (1-4)","Summary":"string (description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Case Importance":"int (1-4)"},
                {"Case Importance":"string (select one of: key_case, 1, 2, 3)"},
                {"Case Importance":"string (Easy or Hard)"},
                {"Case Importance":"string (Easy or Hard)","Summary":"string (description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Case Importance":"string (select one of: key_case, 1, 2, 3)","Summary":"string (description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Summary":"string (description of the case)","Reasoning":"string (give your reason for the importance)","Case Importance":"int (1-4)"},
                {"Court":"string (Judgment or Decision)","Summary":"string (brief description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Court":"string (Judgment or Decision)"},
                {"Court":"string (Committee, Chamber or Grand Chamber)","Summary":"string (brief description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Court":"string (Committee, Chamber or Grand Chamber)"},
                {"Court":"string (Committee or Chamber)","Summary":"string (brief description of the case)","Reasoning":"string (give your reason for the importance)" },
                {"Court":"string (Committee or Chamber)"},
                {"Case Importance":"string (select one of: key_case, 1, 2, 3)","Reasoning":"string (give your reason for the importance)" },
]

PARAMETERS = {'schema':[JSON_SCHEMAS[0],JSON_SCHEMAS[1]],
              'zero_shot':[True,False],
              'text':[1,2,3]
              }

PARAMETERS_BIN = {'schema':[JSON_SCHEMAS[3],JSON_SCHEMAS[4]],
              'zero_shot':[True,False],
              'text':[1,2,3]
              }

PARAMETERS_2 = {'schema':[JSON_SCHEMAS[5]],
              'zero_shot':[False],
              'text':[1,2,3]
              }

PARAMETERS_3 = {'schema':[JSON_SCHEMAS[5],JSON_SCHEMAS[2]],
              'zero_shot':[False],
              'text':[4]
              }

PARAMETERS_4 = {'schema':[JSON_SCHEMAS[13]],
                'zero_shot':[True,False],
                'text':[1,2,3]
                }

PARAMETERS_COURT = {'schema':[JSON_SCHEMAS[7],JSON_SCHEMAS[8]],
              'zero_shot':[True, False],
              'text':[1,2,3]
              }

#parameters for Chamber v Grand Chamber v Committee
PARAMETERS_CHAMBER_1 = {'schema':[JSON_SCHEMAS[9],JSON_SCHEMAS[10]],
              'zero_shot':[True, False],
              'text':[1,2,3]
              }

#parameters for Chamber v Committee
PARAMETERS_CHAMBER_2 = {'schema':[JSON_SCHEMAS[11],JSON_SCHEMAS[12]],
              'zero_shot':[True, False],
              'text':[1,2,3]
              }

JSON_REASONING = {
                  "Reasoning for Key Case":{"Development of Case Law":"The ECtHR noted that it had not previously been called upon to consider whether and to what extent the imposition of a statutory waiting period for granting family reunification to persons who benefit from subsidiary or temporary protection is compatible with Article 8. The ECtHR was thus afforded an opportunity to develop its case law in this respect, concluding that a wide margin of appreciation is afforded, but discretion is not unlimited.","Politically Sensitive":"	The judgment was delivered in the context of states expressing concern regarding the increasing number of asylum-seekers displaced from Syria. Moreover, international bodies had expressed concern regarding Denmark's approach to family reunification. The politically sensitive nature of the case is reflected in the number of third-party observations made."},
                  "Reasoning for Level 1 Case":{"Development/Clarification of Case Law":"•	The ECtHR clarified is approach and the applicable principles to be applied in instances in which no 'properly made claim' for compensation has been made"},
                  "Reasoning for Level 2 Case":{"Reaffirmation of Existing Case Law":"	The ECtHR maintained the conclusions reached in its existing case law concerning the exhaustion of domestic remedied/access to an effective remedy with respect to constitutional redress in the context of conditions of detention.","Development of Case Law":"	By applying existing principles to the specific facts of the applicant's case, to an extent the ECtHR developed the understanding as to what constitutes inhuman and degrading treatment further, specifically with respect to vulnerable who are minors and/or suffering from ill-health.","General Measures":"	In light of the violations found, the ECtHR issued general measures and, notably, stated that it anticipates that the case may give rise to a number of other well-founded applications. For this reason, whilst the case does not significantly contribute to the development of the ECtHR's case law, it nevertheless raises a particular point of interest."},
                  "Reasoning for Level 3 Case":{"Application of Existing Case Law":"In its examination of the merits, the ECtHR applied criteria on expulsion and exclusion orders set out in its previous case law to the facts of the applicant's case. The judgment thus did not develop, but merely applied, its case law. The ECtHR referred to case law that was applied in its judgment in its questions to the parties (i.e., Abdi case). Reference to its earlier case law in Communicated Cases may, therefore, indicate that there are existing authorities for the ECtHR to apply, such that the judgment is likely to be categorised as importance level 3 due to 'simply apply[ing] existing case-law' only. It is notable that the ECtHR did not refer to the Uner or Maslov cases in the Communicated Case, which it cited as setting out the relevant criteria which were then applied in its judgment. In other words, the ECtHR did not cite the cases most pertinent to its judgment at the communication stage."},
                  }

class Experiment_1():

    '''
    Experiment 1 - detecting existing bias concerning cases in the dataset
    Use in run_async to prepare data for to use with the batch API
    '''

    def __init__(self,data,binary=None,grand_chamber=True,reasoning=False):
        self.data = data
        try:
            self.process_data()
        except:
            print('can\'t process data so data accepted unprocessed')
            self.data = data

        self.binary = binary
        self.grand_chamber = grand_chamber
        self.reasoning = reasoning

    def run_async(self, schema:dict = JSON_SCHEMAS[2], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first',temperature:int=0,max_tokens:int=550,top_p=1):
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
            #print(self.data.iloc[file])
            if self.binary == 'binary_difficulty':
                prompt = self.get_binary_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            elif self.binary == 'binary_court':
                prompt = self.get_court_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            elif self.binary == 'chamber_court':
                prompt = self.get_chamber_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info, grand_chamber=self.grand_chamber)
            else:
                prompt = self.get_prompt(self.data.iloc[file], schema, prompt_type=prompt_type, zero_shot=zero_shot, text=text, examples=examples, info=info)
            id = self.data.iloc[file]['Filename']
            template = {"custom_id": f'{id}', "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}],'response_format':{'type': 'json_object'},'max_tokens':max_tokens,'temperature':temperature,'top_p':top_p, 'seed':42}}

            output.append(template)
        
        return output

    def process_data(self):
        metadata = pd.read_json('./Data/overlap_cases/pruned_COMMUNICATEDCASES_meta.json',lines=True)
        metadata.rename(columns={'itemid':'Filename'}, inplace=True)
        self.data = pd.merge(self.data, metadata, on='Filename')
        self.data = self.data[['Filename','importance_x','appno','docname']]
        self.data.rename(columns={'importance_x':'importance'}, inplace=True)

    def get_prompt(self, row,schema:dict = JSON_SCHEMAS[2], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):
        #print(row)
        name = row['docname']
        appnos = row['appno']

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
    
    def get_binary_prompt():
        raise TypeError('Binary prompt not available for Experiment 1')
    def get_court_prompt():
        raise TypeError('Court prompt not available for Experiment 1')
    def get_chamber_prompt():
        raise TypeError('Chamber prompt not available for Experiment 1')
    
class Experiment_2(Experiment_1):

    '''
    Experiment 2 - performing the main experiments of GPT-4o performance across few-shot and zero-shot settings
    '''

    def __init__(self,data,content,binary=False,grand_chamber=False,reasoning=False):
        self.data = data
        try:
            self.process_data()
        except:
            print('can\'t process data so data accepted unprocessed')
            self.data = data
        self.binary = binary
        self.grand_chamber = grand_chamber
        self.reasoning = reasoning

    def process_data(self,content='both'):

        match content:
            case 'questions':
                self.data = self.data[['Filename','Questions']]
            case 'subject_matter':
                self.data = self.data[['Filename','Subject Matter']]
            case 'both':
                self.data = self.data[['Filename','Questions','Subject Matter']]    

    def get_prompt(self, row,schema:dict = JSON_SCHEMAS[2], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):

        '''Function to generate a prompt for the GPT-4o model.
        
        Parameters: 
        row: pd.Series
            A row from the dataframe containing the data.
        zero_shot: bool
            A boolean to determine if the prompt is for zero-shot learning.
        text: int
            The section/s of the text to include in the prompt:
                1 = Subject Matter
                2 = Questions
                3 = Both
        examples: list
            A list of the examples to include in the prompt.
            
        Returns:
        prompt: str
            The prompt to be used for the GPT-4o model.
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
            case 4:
                text = row['Case Summary']
                text_amount = 'case summary'
            case _:
                raise ValueError('Invalid text value. Please enter a value between 1 and 3.')

        if self.reasoning is False:
            if zero_shot:
                additional_context = ''
            else:
                #examples = [f'Importance: {i}\n{e}' for i, e in zip(row['importance'], examples)]
                additional_context = f'''You are also given a number of examples for each level of importance. 
                                        key_case: {examples[0]}; Level 1: {examples[1]}; Level 2: {examples[2]}; Level 3: {examples[3]}'''
        else:
            if zero_shot:
                additional_context = f'You are given a number of examples of how to reason for each level of importance: {JSON_REASONING}'
            else:
                additional_context = f''' You are given a number of paired examples for each level of importance containing the communicated case text and an example of the reasoning which may have led to the importance level:
                                        key_case text: {examples[0]}; key_case reasoning: {JSON_REASONING['Reasoning for Key Case']};
                                        Level 1 text: {examples[1]}; Level 1 reasoning: {JSON_REASONING['Reasoning for Level 1 Case']};
                                        Level 2 text: {examples[2]}; Level 2 reasoning: {JSON_REASONING['Reasoning for Level 2 Case']};
                                        Level 3 text: {examples[3]}; Level 3 reasoning: {JSON_REASONING['Reasoning for Level 3 Case']}
                                        '''

        # importance_levels = '''1: These are the most important and have been selected as key cases and have been selected for publication in the Court\'s official reports; 
        #                     2: The case is of high importance. The case makes a significant contribution to the development, clarification or modification of its case law, either generally or in relation to a particular case; 
        #                     3: The case is of medium importance. The case while not making a significant contribution to the case-law, nevertheless it goes beyond merely applying existing case law; 
        #                     4: The case is of low importance. The case is of limited interest and simply applies existing case law'''
        importance_levels = '''key_case: These are the most important and have been selected as key cases and have been selected for publication in the Court\'s official reports; 
                               1: The case is of high importance. The case makes a significant contribution to the development, clarification or modification of its case law, either generally or in relation to a particular case; 
                               2: The case is of medium importance. The case while not making a significant contribution to the case-law, nevertheless it goes beyond merely applying existing case law; 
                               3: The case is of low importance. The case is of limited interest and simply applies existing case law''''''
                            '''
        if info:
            state_info = 'If you do not know the importance, state that you do not have enough information.'
        else:
            state_info = ''

        prompt = f''' 
        You are a lawyer in the European Court of Human Rights, and your goal is to predict the importance of a case, based on information provided from a communicated case. Importance in a legal setting refers to the significance of a case in terms of its impact on the development of case law.
        The following information is provided to you:
        You will be given a communicated case, including the {text_amount}.
        You are given a description of the different levels of importance: {importance_levels}.
        {additional_context}.
        Based on the information given to you, as well as any relevant case law from the European Court of Human Rights, predict the importance of the case according to the criteria given. 
        {state_info}
        The output should be given directly in JSON format, with the following schema: {schema}.
        The communicated case information you should base your judgement on is as follows: {text}.
        '''
        if self.reasoning:
            prompt += ' Ensure when giving your reason you think through it step by step similarly to the example reasoning provided'

        return prompt
    
    def get_binary_prompt(self, row,schema:dict = JSON_SCHEMAS[3], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):


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
            #examples = [f'Importance: {i}\n{e}' for i, e in zip(row['importance'], examples)]
            additional_context = f'''You are also given a number of examples for each level of importance. 
                                    Easy: {examples[0]}; Hard: {examples[1]}'''

        importance_levels = '''Easy: The case is of limited interest and simply applies existing case law or while not making a significant contribution to the case-law, nevertheless it goes beyond merely applying existing case law;
                               Hard: The case makes a significant contribution to the development, clarification or modification of its case law, either generally or in relation to a particular case or these are the most important and may be selected as a key case and published in the Court\'s official reports.
                               '''
        
        if info:
            state_info = 'If you do not know the importance, state that you do not have enough information.'
        else:
            state_info = ''

        prompt = f''' 
        You are a lawyer in the European Court of Human Rights, and your goal is to predict the importance of a case, based on information provided from a communicated case. Importance in a legal setting refers to the significance of a case in terms of its impact on the development of case law.
        The following information is provided to you:
        You will be given a communicated case, including the {text_amount}.
        You are given a description of the different levels of importance: {importance_levels}.
        {additional_context}.
        Based only on the information given to you predict the importance of the case according to the criteria given, giving a response of either 1, 2, 3 or 4. 
        {state_info}
        The output should be given directly in JSON format, with the following schema: {schema}.
        The communicated case information you should base your judgement on is as follows: {text}.
        '''

        return prompt

    def get_court_prompt(self, row,schema:dict = JSON_SCHEMAS[3], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first'):

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
            #examples = [f'Importance: {i}\n{e}' for i, e in zip(row['importance'], examples)]
            additional_context = f'''You are also given a number of examples for each level of the court. 
                                    Judgment: {examples[0]}; Decision: {examples[1]}'''

        importance_levels = '''Judgment: In a judgment, the European Court of Human Rights will rule on the merits and/or just satisfaction, possibly in addition to the admissibility, of a complaint. The European Court of Human Rights may also deliver a judgment striking out an application. This may be classified as a judgment if done at a late stage in proceedings (e.g., if the European Court of Human Rights has previously declared the application admissible or if it has issued a judgment but reserved the question of the application of Article 41 ECHR). ;
                               Decision: In a decision, the European Court of Human Rights will confine its examination to the admissibility of an application only. Thus, if an application is declared inadmissible, the European Court of Human Rights will issue a decision to that effect. However, if an application is declared admissible, the European Court of Human Rights will proceed to examine the merits and deliver a judgment -as opposed to a decision- accordingly. The European Court of Human Rights may also issue a decision striking out an application.
                            '''
        
        if info:
            state_info = 'If you do not know the importance, state that you do not have enough information.'
        else:
            state_info = ''

        prompt = f''' 
        You are a lawyer in the European Court of Human Rights, and your goal is to predict whether a case will end up as a Judgment or a Decision, based on information provided from a communicated case. 
        The following information is provided to you:
        You will be given a communicated case, including the {text_amount}.
        You are given a description of the different levels of the court: {importance_levels}.
        {additional_context}.
        Based only on the information given to you predict the importance of the case according to the criteria given, giving a response of either Judgment or Decision. 
        {state_info}
        The output should be given directly in JSON format, with the following schema: {schema}.
        The communicated case information you should base your outcome on is as follows: {text}.
        '''

        return prompt

    def get_chamber_prompt(self, row,schema:dict = JSON_SCHEMAS[3], zero_shot:bool =True, text:int = 3, examples:list = [], info:bool=True, prompt_type:str='first',grand_chamber=False):
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
            #examples = [f'Importance: {i}\n{e}' for i, e in zip(row['importance'], examples)]
            if grand_chamber:
                additional_context = f'''You are also given a number of examples for each level of the court. 
                                    Committee: {examples[0]}; Chamber: {examples[1]}; Grand Chamber: {examples[2]}'''
            else:
                additional_context = f'''You are also given a number of examples for each level of the court.
                                         Committee: {examples[0]}; Chamber: {examples[1]}'''
        
# source for information = https://www.berger-avocat.eu/en/ecthr/structure-and-functioning-of-the-ecthr
        if grand_chamber:
            importance_levels = '''Committee: A Committee consists of 3 judges. The Committee can rule on the merits of a case where the Court's case law is well established. They can also rule on the admissibility of a case with well established case law.; 
                                   Chamber: A Chamber consists of 7 judges. The Chamber can can decide on the merits or admissibility of a case if no prior decision has been reached by the Committee or a single judge. The Chamber can relinquish jurisdiction to the Grand Chamber.;
                                   Grand Chamber: The Grand Chamber consists of 17 judges. examines the cases that are submitted to it either after a Chamber has relinquished jurisdiction or when a request for referral of the case has been granted by the Grand Chamber Panel. The Grand Chambers deals with cases which raise a serious question affecting the interpretation of the Convention or if there is a risk that its resolution of the case would be inconsistent with a judgment previously delivered by the Court. 
                                '''
            level = 'Committee, Chamber or Grand Chamber'
        else:
            importance_levels = '''Committee: A Committee consists of 3 judges. The Committee can rule on the merits of a case where the Court's case law is well established. They can also rule on the admissibility of a case with well established case law.  
                                   Chamber: Here we consider both the Chamber and Grand Chamber. A Chamber consists of 7 judges and the Grand Chamber consists of 17 judges. The Chamber can can decide on the merits or admissibility of a case if no prior decision has been reached by the Committee or a single judge. The Chamber can relinquish jurisdiction to the Grand Chamber which deals with cases which raise a serious question affecting the interpretation of the Convention or if there is a risk that its resolution of the case would be inconsistent with a judgment previously delivered by the Court.
                                '''
            level = 'Committee or Chamber'
        
        if info:
            state_info = 'If you do not know the importance, state that you do not have enough information.'
        else:
            state_info = ''

        prompt = f''' 
        You are a lawyer in the European Court of Human Rights, and your goal is to predict whether a case will end up at the level of {level}; based on information provided from a communicated case. 
        The following information is provided to you:
        You will be given a communicated case, including the {text_amount}.
        You are given a description of the different levels of the court: {importance_levels}.
        {additional_context}.
        Based only on the information given to you predict the importance of the case according to the criteria given, giving a response of either Judgment or Decision. 
        {state_info}
        The output should be given directly in JSON format, with the following schema: {schema}.
        The communicated case information you should base your outcome on is as follows: {text}.
        '''

        return prompt

def create_examples(data,text,importance = True):
    '''Function to create examples for the few-shot learning prompt.
    '''
    examples = []

    if importance:
        data.sort_values(by='importance', inplace=True)
    else:
        pass        

    for i in range(len(data)):
        match text:
            case 1:
                example = data.iloc[i]['Subject Matter']
            case 2:
                example = data.iloc[i]['Questions']
            case 3:
                example = data.iloc[i]['Subject Matter'] + ' ' + data.iloc[i]['Questions']
            case 4:
                example = data.iloc[i]['Case Summary']

        examples.append(example)
                
    return examples

def generate_example_candidates(data,keyword,labels):
    example_candidates = pd.DataFrame(columns=data.columns)
    for i in range(len(labels)):
        row = data[data[keyword]==labels[i]].sample(1, random_state=42,axis=0)
        example_candidates = pd.concat([example_candidates,row],axis=0)
    #print(example_candidates)
    return example_candidates

def save_file(output,filepath,batch_name,prompt_type='first',experiment=1, zero_shot=False,text=3,schema=JSON_SCHEMAS[0], identifier=None):
    if identifier is not None:
        identifier = identifier
    else:
        identifier = ''

    if experiment == 1:
        with open(f'{filepath}/{batch_name}_{prompt_type}.jsonl', 'w') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')
    else:
        with open(f'{filepath}/{batch_name}_{zero_shot}_{text}_{JSON_SCHEMAS.index(schema)}_{identifier}.jsonl', 'w') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    
    client = OpenAI(api_key=openai_key)
    df = pd.read_pickle('valid_data.pkl')    

    #CHAMBER EXPERIMENTS
    #court_label_data = pd.read_pickle('Chamber_Data_2.pkl')
    #pd.set_option('display.max_columns', None)
    #df = pd.merge(df, court_label_data, on=['Filename','Questions','Subject Matter','importance'],how='left')
    #df_example = generate_example_candidates(df,keyword='Court',labels =['Committee','Chamber'])

    #print(df_main)
    df_main = df[:-4]
    df_example = df[-4:]
    #print(df_example)

    #df_main = df.drop(df_example.index)

    #print(len(df_main))
  
    # ASYNC EXAMPLE - EXPERIMENT 2
    exp_2 = Experiment_2(df,content='both',grand_chamber=False,reasoning=True)

    #set param grid search
    grid = model_selection.ParameterGrid(PARAMETERS_4)

    for params in grid:
        #set examples
        examples = create_examples(df_example,text=params['text'],importance=False)
        output = exp_2.run_async( schema=params['schema'],zero_shot=params['zero_shot'],text=params['text'],examples=examples,info=True,temperature=0,max_tokens=550)
        save_file(output,filepath='./Batches/experiment_2_CoT',batch_name='experiment_2_CoT',experiment = 2, zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'])

    # ASYNC 

    # for i in range(20):

    #     sampled_df = df.groupby('importance', group_keys=False).apply(lambda x: x.sample(min(len(x), 1), random_state=i))
    #     df_main = df.drop(sampled_df.index)

    #     exp_2 = Experiment_2(df_main, content='both', binary=False)

    #     #set param grid search
    #     grid = model_selection.ParameterGrid(PARAMETERS_2)

    #     for params in grid:
    #         #set examples
    #         examples = create_examples(sampled_df,text=params['text'])
    #         #easy = examples[3]
    #         #hard = examples[0]
    #         #example_list = [easy,hard]
    #         output = exp_2.run_async(schema=params['schema'],zero_shot=params['zero_shot'],text=params['text'],examples=examples,info=True,temperature=0,max_tokens=550)
    #         save_file(output,filepath='./Batches/experiment_sample',batch_name='experiment_sample',experiment = 2, zero_shot=params['zero_shot'],text=params['text'],schema=params['schema'], identifier=i)

    # ASYNC EXAMPLE - EXPERIMENT 1
    #exp_1 = Experiment_1(df_main)
    #for num in ['first','second','third','fourth','fifth']:
    #    exp_1.run_async(filepath='./Batches/experiment_1_valid',batch_name='experiment_1',schema=JSON_SCHEMAS[2],prompt_type='first',temperature=0,max_tokens=500)  

    #submit files to BATCH API
    
    # SYNC EXAMPLE
    # output = exp_1.run_sync(JSON_SCHEMAS[0],'first',temperature=0,max_tokens=500)
    # output.to_pickle('exp_1_output.pkl')
    # print(output)


