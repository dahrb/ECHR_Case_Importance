from transformers import AutoModel, AutoTokenizer
import transformers
import torch
import os
from API_key import hugging_face_key
#from accelerate import Accelerator
import time
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

#UPDATE TRANSFORMERS LIBRARY!!! DO!!!!!!!!!!!!4



# model_id = "/users/sgdbareh/volatile/ECHR_Importance/models/llama-test-abc"
# # #quantization_config = bnb.BitsAndBytesConfig(load_in_8bit=True)
#model_id = "unsloth/Llama-3.3-70B-Instruct-GGUF"

# from transformers import AutoTokenizer, AutoModelForCausalLM

# #model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# filename = "Llama-3.3-70B-Instruct-Q2_K.gguf"

# tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
# model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)

#model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)#,quantization_config=quantization_config)
# #model.gradient_checkpointing_enable()
# #model = accelerator.prepare(model)
# #tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.3-70B-Instruct-GGUF", device_map="auto")

# #model = AutoModel.from_pretrained("unsloth/Llama-3.3-70B-Instruct-GGUF")  

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map= 'auto',
#     token=hugging_face_key
# )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# outputs = pipeline(
#         messages,
#         max_new_tokens=256,
#         temperature=0,
#         top_p=1
#     )

# print(outputs[0]["generated_text"][-1])

#check hyperparameters
# super_start = time.time()
# for i in range(100):
#     start_time = time.time()
#     print(i)
#     outputs = pipeline(
#         messages,
#         max_new_tokens=256,
#         temperature=0,
#         top_p=1
#     )
#     print(outputs[0]["generated_text"][-1])
#     end_time = time.time()
#     print(f"Execution time {i}: {end_time - start_time} seconds")

# super_end = time.time()
# print(f"Total execution time: {super_end - super_start} seconds")

# ## Imports
# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama

# ## Download the GGUF model
# model_name = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
# model_file = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf" # this is the specific model file we'll use in this example. It's a 4-bit quant, but other levels of quantization are available in the model repo if preferred
# model_path = hf_hub_download(model_name, filename=model_file)

# ## Instantiate model from downloaded file
# llm = Llama(
#     model_path=model_path,
#     n_ctx=16000,  # Context length to use
#     n_threads=32,            # Number of CPU threads to use
#     n_gpu_layers=0        # Number of model layers to offload to GPU
# )

# ## Generation kwargs
# generation_kwargs = {
#     "max_tokens":20000,
#     "stop":["</s>"],
#     "echo":False, # Echo the prompt in the output
#     "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
# }

# ## Run inference
# prompt = "The meaning of life is "
# res = llm(prompt, **generation_kwargs) # Res is a dictionary

# ## Unpack and the generated text from the LLM response dictionary and print it
# print(res["choices"][0]["text"])
# # res is short for result

# from gguf import GGUFModel
# from transformers import AutoModel, AutoTokenizer

# # Load your tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# gguf_model = GGUFModel("/users/sgdbareh/volatile/ECHR_Importance/models/llama-test-abc/Llama-3.3-70B-Instruct-Q2_K.gguf")
# input_text = "Your input text here"
# inputs = tokenizer(input_text, return_tensors="pt")
# outputs = gguf_model(**inputs)
# print(outputs)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import bitsandbytes as bnb

# # Set the model ID and API token
# model_id = "unsloth/Llama-3.3-70B-Instruct-GGUF"
# api_token = "your_hugging_face_api_token_here"

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=api_token)

# # Load the model with 8-bit quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     load_in_8bit=True,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     use_auth_token=api_token
# )

# # Check if bitsandbytes is working by performing a simple inference
# def check_bitsandbytes():
#     try:
#         # Prepare a simple input
#         input_text = "Hello, how are you?"
#         inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

#         # Perform inference
#         outputs = model.generate(**inputs)
#         output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         print("Bitsandbytes is working correctly.")
#         print("Generated text:", output_text)
#     except Exception as e:
#         print("An error occurred:", e)

# if __name__ == "__main__":
#     check_bitsandbytes()

#TEST 4-BIT - DONE
#TEST 4-BIT ACCELERATE
#TEST 8-BIT ACCELERATE - IN PROGRESS 
#TEST 16-BIT ???
#TEST 32-BIT ???
#TEST SPEEDS OF MULTIPLE PROMPTS
#TEST JSON FORMATS
#GET OTHER LLM 

#model_id = "meta-llama/Llama-3.3-70B-Instruct"
#model_id = "MaziyarPanahi/calme-3.2-instruct-78b"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
accelerator = Accelerator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#accelerator = Accelerator()
device = accelerator.device

print(f"Accelerator device: {device}")

# Print the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")



quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,torch_dtype=torch.bfloat16, device_map={'':torch.cuda.current_device()},#device_map='auto',
 quantization_config=quantization_config)

    #quantized_model.gradient_checkpointing_enable()
    
quantized_model = accelerator.prepare(quantized_model)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print the device for each part of the model
for name, param in quantized_model.named_parameters():
    print(f"Parameter {name} is on device {param.device}")

for i in range(10):
    input_text = f"What are we having for dinner {i}?"
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)

    output = quantized_model.module.generate(**input_ids, max_new_tokens=10)

    print(tokenizer.decode(output[0], skip_special_tokens=True))


# from llama_cpp import Llama

# llm = Llama.from_pretrained(
#     repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
#     filename="*q8_0.gguf",
#     verbose=False,
#     seed=42,
#     n_gpu_layers=10
# )

# output = llm(
#       "Q: Name the planets in the solar system? A: ", # Prompt
#       max_tokens=150, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=False # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion
# print(output)