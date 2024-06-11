import pandas as pd
import tiktoken


#Source - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(data, zero_shot = True) -> int:
    """Returns the number of tokens in a text string."""

    subject_matter = data['subject_matter'].values
    questions = data['questions'].values

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

data = pd.read_pickle('Data_1_0.pkl')

num_tokens_from_string(data)
num_tokens_from_string(data,zero_shot=False)


#len(gpt_4.encode('hello world!'))