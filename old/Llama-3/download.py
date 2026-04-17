from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from statistics import mean
import torch, time, json
from API_key import hugging_face_key
from huggingface_hub import hf_hub_download


model_path = "Qwen/Qwen2.5-72B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.bfloat16,
    token=hugging_face_key
)
print('Done')