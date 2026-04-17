import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import logging
import math
from torch.utils.data import DataLoader, Subset
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from utils import EarlyStopping, calculate_mcc, custom_collate_fn, CrossEncoderMod
from torch.amp import GradScaler, autocast
from optparse import OptionParser

torch.cuda.empty_cache()

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)
#### /print debug information to stdout

parser = OptionParser(usage='usage: -l learning_rate -b batch_size -d dropout')   
parser.add_option("-d", "--dropout", action = "store", dest='dropout', type = "float", default = 0.0)
parser.add_option("-l", "--learning_rate", action = "store", dest='learning_rate', type = "float", default = 1e-5)
parser.add_option("-b", "--batch_size", action = "store", dest='batch_size', type = "int", default = 16)
(options, _) = parser.parse_args()

dropout = options.dropout
learning_rate = options.learning_rate
batch_size = options.batch_size

model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define our Cross-Encoder
#train_batch_size = 1
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

# Split the data into training and test sets
train_samples, test_samples = train_test_split(train_samples, test_size=0.2, random_state=456)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=198)

# Hyperparameter grid
param_grid = {
    'learning_rate': [learning_rate],
    'batch_size': [batch_size],
    'dropout': [dropout]
}

# List to store results for each combination of hyperparameters
results = []

# Perform hyperparameter tuning
best_ap_score = 0
best_params = None

for lr in param_grid['learning_rate']:
    for bs in param_grid['batch_size']:
        for dropout in param_grid['dropout']:
            logger.info(f"Training with lr={lr}, batch_size={bs}, dropout={dropout}")

            fold_results = []

            for fold, (train_index, val_index) in enumerate(kf.split(train_samples)):
                 
                logger.info(f"Training fold {fold + 1}")

                # Initialize the model with dropout
                model = CrossEncoderMod(model_name, num_labels=1, classifier_dropout=dropout)

                # Create subset dataloaders for the current fold
                train_subset = Subset(train_samples, train_index)
                val_subset = Subset(train_samples, val_index)
    
                train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=bs)#,collate_fn=lambda x: model.smart_batching_collate(x))
                #val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=bs,collate_fn=custom_collate_fn)

                # Configure the training
                warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # Warm-up steps
                logger.info(f"Warmup-steps: {warmup_steps}")
                
                # Define evaluator for the validation set
                evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_subset, name='Relevance BERT')

                # Train the model
                model.fit(
                    train_dataloader=train_dataloader,
                    evaluator=evaluator,
                    epochs=num_epochs,
                    evaluation_steps=0,
                    optimizer_params={'lr': lr},
                    warmup_steps=warmup_steps,
                    output_path=f"{model_save_path}_fold_{fold + 1}"
                )

                # Evaluate the model on the validation set
                evaluation_result = evaluator(model)
                logger.info(f"Fold {fold + 1} evaluation result: {evaluation_result}")
                
                # Store the best model for the current fold
                fold_results.append(evaluation_result)

            # Calculate the average F1 score across all folds
            avg_ap_score = sum(fold_results) / len(fold_results)
            logger.info(f"Average AP score for lr={lr}, batch_size={bs}, dropout={dropout}: {avg_ap_score}")

            # Update best hyperparameters if current combination is better
            if avg_ap_score > best_ap_score:
                best_ap_score = avg_ap_score
                best_params = {
                    'learning_rate': lr,
                    'batch_size': bs,
                    'dropout': dropout,
                    'epochs':model.epoch
                }

logger.info(f"Hyperparameters: {best_params} and avg AP {best_ap_score}")

with open("results.txt", "a") as myfile:
    myfile.write(f"Hyperparameters: {best_params} and avg AP {best_ap_score} \n")



