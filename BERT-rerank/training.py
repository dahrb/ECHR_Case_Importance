import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from datetime import datetime
from zipfile import ZipFile
import logging
import math
from torch.utils.data import DataLoader, Subset
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import torch
from torch.optim import AdamW
from sklearn.model_selection import KFold
torch.cuda.empty_cache()

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout

model_name = "nlpaueb/legal-bert-base-uncased"
model = CrossEncoder(model_name,num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Define our Cross-Encoder
train_batch_size = 4
num_epochs = 30
model_save_path = "/users/sgdbareh/volatile/ECHR_Importance/BERT-rerank/model" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
use_cuda = torch.cuda.is_available()

train = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/BERT-rerank/BERT_training_data_df.pkl')

label2int = {"neg": 0, "pos": 1}

train_samples = []

# Iterate over the DataFrame rows
for index, row in train.iterrows():
    # Create InputExample for positive label
    train_samples.append(InputExample(texts=[str(row['Comm_Case']).lower(), str(row['Positive']).lower()], label=label2int["pos"]))
    # Create InputExample for negative label
    train_samples.append(InputExample(texts=[str(row['Comm_Case']).lower(), str(row['Negative']).lower()], label=label2int["neg"]))

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List to store results for each fold
results = []

for fold, (train_index, val_index) in enumerate(kf.split(train_samples)):
    logger.info(f"Training fold {fold + 1}")

    # Create subset dataloaders for the current fold
    train_subset = Subset(train_samples, train_index)
    val_subset = Subset(train_samples, val_index)

    train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=train_batch_size)
    #val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=train_batch_size)

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info(f"Warmup-steps: {warmup_steps}")
    
    # Define evaluator for the validation set
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_subset,name='Relevance BERT')

    # Initialize the model
    model = CrossEncoder(model_name, num_labels=1)

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=f"{model_save_path}_fold_{fold + 1}"
    )

    # Evaluate the model on the validation set
    evaluation_result = evaluator(model)
    logger.info(f"Fold {fold + 1} evaluation result: {evaluation_result}")
    
    # Store the result
    results.append({
        'fold': fold + 1,
        'evaluation_result': evaluation_result
    })

# Print all results
for result in results:
    print(f"Fold {result['fold']} evaluation result: {result['evaluation_result']}")

results_df = pd.DataFrame(results)
results_df.to_pickle("/users/sgdbareh/volatile/ECHR_Importance/BERT-rerank/results_df.pkl")