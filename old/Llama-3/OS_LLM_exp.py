from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch, time
from API_key import hugging_face_key
import pandas as pd
import sys
import os
import json
from transformers import set_seed
from optparse import OptionParser
import datetime
import re
torch.distributed.init_process_group(init_method=None, timeout=datetime.timedelta(seconds=1800), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)
set_seed(42)

sys.path.insert(0, '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION')

accelerator = Accelerator()

parser = OptionParser(usage='usage: ')   
parser.add_option("-k", "--K", action = "store", type = "int", dest = "k", default = 3)
parser.add_option("-e", "--embedding_name", action = "store", type = "string", dest = "embedding_name")
parser.add_option("-m", "--model_path", action = "store", type = "string", dest = "model_path")
parser.add_option("-f", "--filepath", action = "store", type = "string", dest = "filepath")

(options, _) = parser.parse_args()

from transformers import StoppingCriteria, StoppingCriteriaList

# Define a stopping criteria for triple backticks
class StopOnTripleBackticks(StoppingCriteria):
    def __init__(self, tokenizer):
        # Encode "```" without special tokens
        self.stop_tokens = tokenizer.encode("```", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # If there's not enough tokens yet, keep going
        if len(input_ids[0]) < len(self.stop_tokens):
            return False
        # Compare last few tokens to the encoded backticks
        if torch.equal(input_ids[0][-len(self.stop_tokens):], torch.tensor(self.stop_tokens, device=input_ids.device)):
            return True
        return False

#FILE ARGS HERE
FILEPATH = options.filepath #'/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/TEST_SEMANTIC_NO_RELEVANCE'
K = options.k
EMBEDDING = options.embedding_name
MODEL_PATH = options.model_path #"Qwen/Qwen2.5-72B-Instruct"

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
            
print('111111')
        
quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)#,bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,    
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    token=hugging_face_key
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)   

stop_criteria = StoppingCriteriaList([StopOnTripleBackticks(tokenizer)])

model.gradient_checkpointing_enable()

# # Function to validate JSON
# def is_valid_json(output):
#     try:
#         json.loads(output)
#         return True
#     except ValueError:
#         return False

# store output of generations in dict
results=dict(outputs=[], num_tokens=0)

counter = 0

with accelerator.split_between_processes(data) as prompts:

    ### have each GPU do inference, prompt by prompt
    for prompt in prompts:
        
        print(f'Counter: {counter}')
        
        prompt = f"Only respond in structured JSON format following the schema given below, do not generate any additional text under any circumstance: \n{prompt} ```json"
        
        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=700,temperature=1e-5,do_sample=True,top_p=1,top_k=1, stopping_criteria=stop_criteria)[0]
        
        output = tokenizer.decode(output_tokenized, skip_special_tokens=True)
        
        # Extract and validate JSON output
        output = output[len(prompt):].strip()

        # # A simple regex to detect if a string starts and ends with JSON braces or brackets:
        # json_pattern = re.compile(r'^\s*(\{.*\}|\[.*\])\s*$', re.DOTALL)

        # match = json_pattern.match(output)
        
        # if match:
        #     print("Possible JSON structure found")
        # else:
        #     print("No valid JSON structure found")
            
        results["outputs"].append(output)
        results["num_tokens"] += len(output)
        
        # if is_valid_json(output):
        #     #print(f"Valid JSON output: {counter}")
        #     #print('+++++++++++++++++++++++++++++')
        #     results["outputs"].append(output)
        #     results["num_tokens"] += len(output)
        # else:
        #     #print(f"Invalid JSON output: {output}")
        #     #print('-----------------------------')
        #     try:
        #         output = output.split("```json")
        #         output = "```json" + output[1]
        #         #store outputs and number of tokens in result{}
        #         results["outputs"].append(output.strip('```json\n').strip('```'))        
            
        #         results["num_tokens"] += len(output)
        #     except:
        #         print('super fail')
                    
        #print(output)
        counter += 1
        

results=[results] # transform to list, otherwise gather_object() will not collect correctly

# Wait for all processes to finish
accelerator.wait_for_everyone()

# collect results from all the GPUs
results_gathered=gather_object(results)

if accelerator.is_main_process:
    output_dir = '/users/sgdbareh/volatile/ECHR_Importance/Llama-3/Results'
    os.makedirs(output_dir, exist_ok=True)
    
    MODEL_PATH = MODEL_PATH.split('/')[-1]
    
    if FILEPATH == '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/TEST_SEMANTIC_NO_RELEVANCE' or FILEPATH == '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/KG_NO_SEM_TEST/test':
        RELEVANCE = False
    else:
        RELEVANCE = True
        
    if 'KG' in FILEPATH:
        KG_FLAG = 'KG'
    else:
        KG_FLAG = 'NO_KG'
        
    if 'GOLD' in FILEPATH:
        KG_FLAG = 'GOLD'
    else:
        pass
    
    if 'baseline' in FILEPATH:
        KG_FLAG = 'baseline'
    else:
        pass
    
    if 'Court' in FILEPATH:
        KG_FLAG = 'Court'
    else:
        pass
        
    output_file = os.path.join(output_dir, f'results_{RELEVANCE}_{K}_{EMBEDDING}_{MODEL_PATH}_{KG_FLAG}.jsonl')
    
    with open(output_file, 'w') as f:
        for result in results_gathered:
            for output in result["outputs"]:
                json.dump({"output": output}, f)
                f.write('\n')
    
    print(f"Results saved to {output_file}")




