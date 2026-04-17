from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, setup_chat_format
from datasets import load_dataset

import torch
from accelerate import Accelerator

accelerator = Accelerator()

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)


# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct',
                                             device_map={'':torch.cuda.current_device()},
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=quantization_config,
                                             )
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')


# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank adaptation
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none"  # Whether to adapt bias terms
)

model = get_peft_model(model, lora_config)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

with accelerator.main_process_first():

    train_dataset = load_dataset('json', data_files='/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/FINE-TUNE/llama_train_data.jsonl')
    eval_dataset = load_dataset('json', data_files='/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/FINE-TUNE/llama_test_data.jsonl')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = {
    "output_dir": "/users/sgdbareh/volatile/ECHR_Importance/Llama-3/output",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 5,
    "gradient_accumulation_steps": 2,
    "learning_rate": 0.1,
    "logging_dir": "/users/sgdbareh/volatile/ECHR_Importance/Llama-3/output/logs",
    "logging_steps": 100,

}

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        learning_rate=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=42,
    ),
    train_dataset=train_dataset['train'],
    eval_dataset=eval_dataset['train'],
    data_collator=data_collator,
)

# # Prepare everything with accelerator
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader())
# )
# Fine-tune the model using Accelerate
for epoch in range(training_args["num_train_epochs"]):
    model.train()
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    # Evaluate the model at the end of each epoch
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-llama-lora')
tokenizer.save_pretrained('./fine-tuned-llama-lora')