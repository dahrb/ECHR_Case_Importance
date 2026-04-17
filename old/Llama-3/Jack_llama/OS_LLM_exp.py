from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch, time
#from API_key import hugging_face_key
import pandas as pd
import sys
import os
import json
from transformers import set_seed
from optparse import OptionParser

set_seed(42)

accelerator = Accelerator()

parser = OptionParser(usage='usage: ')   
parser.add_option("-k", "--K", action = "store", type = "int", dest = "k", default = 3)
parser.add_option("-e", "--embedding_name", action = "store", type = "string", dest = "embedding_name")
parser.add_option("-m", "--model_path", action = "store", type = "string", dest = "model_path")
parser.add_option("-f", "--filepath", action = "store", type = "string", dest = "filepath")

(options, _) = parser.parse_args()

#FILE ARGS HERE
FILEPATH = options.filepath #'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/TEST_SEMANTIC_NO_RELEVANCE'
K = options.k
EMBEDDING = options.embedding_name
MODEL_PATH = options.model_path #"Qwen/Qwen2.5-72B-Instruct"

#Load data on main process
with accelerator.main_process_first():
    jsonl_files = [f for f in os.listdir(FILEPATH) if f.endswith('.jsonl')]

    file = next((f for f in jsonl_files if f'{K}' in f and EMBEDDING in f), None)
    if not file:
        raise FileNotFoundError(f"No file found with K{K} and {EMBEDDING} in the name")

    data = []
    with open(os.path.join(FILEPATH, file), 'r') as f:
        for line in f:
            json_line = json.loads(line)
            body = json_line.get('body', '')
            messages = body.get('messages', [])
            content = messages[0].get('content', '')
            content = content.strip()
            data.append(content)

#load model
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,    
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    #token=hugging_face_key
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)   

#splits computation across available GPUs
with accelerator.split_between_processes(data) as prompts:
    # store output of generations in dict
    results=dict(outputs=[], num_tokens=0)

    # have each GPU do inference, prompt by prompt
    for prompt in prompts:
        
        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=1000,temperature=1e-5,do_sample=True,top_p=1,top_k=1)[0]
        
        output = tokenizer.decode(output_tokenized, skip_special_tokens=True)
        
        output=output[len(prompt):].strip()
        
        output = output.split("```json")
        
        output = "```json" + output[1]

        #store outputs and number of tokens in result{}
        results["outputs"].append(output.strip('```json\n').strip('```'))
                
      
        results["num_tokens"] += len(output)

    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
results_gathered=gather_object(results)

#collate output on main process
if accelerator.is_main_process:
    output_dir = '/users/sgdbareh/volatile/ECHR_Importance/Llama-3/Results'
    os.makedirs(output_dir, exist_ok=True)
    
    MODEL_PATH = MODEL_PATH.split('/')[-1]
    
    if FILEPATH == '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/TEST_SEMANTIC_NO_RELEVANCE':
        RELEVANCE = False
    else:
        RELEVANCE = True
        
    output_file = os.path.join(output_dir, f'results_{RELEVANCE}_{K}_{EMBEDDING}_{MODEL_PATH}.jsonl')
    
    with open(output_file, 'w') as f:
        for result in results_gathered:
            for output in result["outputs"]:
                json.dump({"output": output}, f)
                f.write('\n')
    
    print(f"Results saved to {output_file}")




