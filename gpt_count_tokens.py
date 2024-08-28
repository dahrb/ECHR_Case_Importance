"""
Version history
v1_1 = provides estimate for JSONL file required for batch api
v1_0 = provides estimate for gpt-4 costs from the dataset under certain assumptions
"""


import pandas as pd
import tiktoken
import os
import os

#Starting point for code is this guide - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(data, zero_shot = True) -> int:
    """Returns the number of tokens in a text string."""

    subject_matter = data['Subject Matter'].values
    questions = data['Questions'].values

    gpt_4 = tiktoken.encoding_for_model('gpt-4')
    gpt_4_o = tiktoken.encoding_for_model('gpt-4o')

    embeddings = ['gpt_4', 'gpt_4_o']
    
    for embed in embeddings:

        if embed == 'gpt_4':
            model = gpt_4
        elif embed == 'gpt_4_o':
            model = gpt_4_o

        num_tokens = 0
        
        for i in zip(subject_matter, questions):
            #subject matter tokens
            num_tokens += len(model.encode(i[0]))
            #question tokens
            num_tokens += len(model.encode(i[1]))
            num_tokens += 3 #to account for system messages
            if zero_shot is True:
                num_tokens += 300
            else:
                num_tokens += 3000
        
        output_tokens = 500*len(subject_matter)
        print(output_tokens,'and',len(subject_matter))

        #calc cost
        if embed == 'gpt_4':
            cost_input =  (num_tokens/1_000_000) * 10 
            cost_output = (output_tokens/1_000_000) * 30
            
            if zero_shot:
                ablation_cost = ((num_tokens + (300*len(data)))/1_000_000)*10 + cost_output*2
            else:
                ablation_cost = ((num_tokens + (3000*len(data)))/1_000_000)*10 + cost_output*2

        elif embed == 'gpt_4_o':
            cost_input =  (num_tokens/1_000_000) * 5 
            cost_output = (output_tokens/1_000_000) * 15

            if zero_shot:
                ablation_cost = ((num_tokens + (300*len(data)))/1_000_000)*5 + cost_output*2
            else:
                ablation_cost = ((num_tokens + (3000*len(data)))/1_000_000)*5 + cost_output*2

        print(f"Number of input tokens in the text: {num_tokens}; embedding used: {embed}; estimated cost: ${cost_input/2:.2f} and £{(cost_input/0.78)/2:.2f}")
        print(f"Number of output tokens (estimate): {output_tokens}; estimated cost: ${cost_output/2:.2f} and £{(cost_output/0.78)/2:.2f}")
        print(f'Total Cost {embed}; zero-shot {zero_shot}: ${(cost_input + cost_output)/2:.2f} and £{((cost_input + cost_output)/0.78)/2:.2f}')
        print(f'Ablation Total cost ${ablation_cost/2:.2f} and £{(ablation_cost/0.78)/2:.2f}')

def num_tokens_from_jsonl(data,name,output_tokens = 15):
    ''''
    Gives price estimation of JSONL file for batch API
    '''
    total_tokens = 0
    tokens_per_message = 3
    tokens_per_name = 1
    gpt_4_o = tiktoken.encoding_for_model('gpt-4o')

    for i in data['body']:
        message = i['messages']
        total_tokens += tokens_per_message
        
        for key, value in message[0].items():
            total_tokens += len(gpt_4_o.encode(value))
            if key == "name":
                total_tokens += tokens_per_name
        total_tokens += 3 
    
    cost_input =  (total_tokens/1_000_000) * 5 

    output_tokens = output_tokens*len(data)
    cost_output = (output_tokens/1_000_000) * 15

    print(f'{name}:\n')
    print(f"Number of input tokens in the text: {total_tokens}; estimated cost: ${cost_input/2:.2f} and £{(cost_input*0.78)/2:.2f}")
    print(f"Number of output tokens (estimate): {output_tokens}; estimated cost: ${cost_output/2:.2f} and £{(cost_output*0.78)/2:.2f}")
    
    total_cost = (cost_input*0.78)/2 + (cost_output*0.78)/2
    print(f'Total Cost: ${total_cost:.2f} and £{total_cost*0.78:.2f}\n')
    round(total_cost,2)

    return total_cost
    
if __name__ == '__main__':

    #num_tokens_from_string EXAMPLE
    #data = pd.read_pickle('valid_data.pkl')
    #print(data)
    #num_tokens_from_string(data)
    #num_tokens_from_string(data,zero_shot=False)

    filename = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches'

    #num_tokens_from_jsonl EXAMPLE
    total_cost = 0
    for i in os.listdir(filename):
        if i.endswith('.jsonl'):
            data = pd.read_json(f'{filename}/{i}',lines=True)
            cost = num_tokens_from_jsonl(data,name=i,output_tokens=500)
            total_cost += cost
    print(f'Total Cost: ${total_cost:.2f} and £{total_cost*0.78:.2f}\n')


