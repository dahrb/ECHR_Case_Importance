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
parser.add_option("-e", "--epochs", action = "store", dest='epochs', type = "int", default = 30)

(options, _) = parser.parse_args()

dropout = options.dropout
learning_rate = options.learning_rate
batch_size = options.batch_size
num_epochs = options.epochs

model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define our Cross-Encoder
#train_batch_size = 1

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

# Hyperparameter grid
param_grid = {
    'learning_rate': [learning_rate],
    'batch_size': [batch_size],
    'dropout': [dropout]
}

# # List to store results for each combination of hyperparameters
# results = []

# # Perform hyperparameter tuning
# best_ap_score = 0
# best_params = None

# for lr in param_grid['learning_rate']:
#     for bs in param_grid['batch_size']:
#         for dropout in param_grid['dropout']:
#             logger.info(f"Training with lr={lr}, batch_size={bs}, dropout={dropout}")

#             # Initialize the model with dropout
#             model = CrossEncoderMod(model_name, num_labels=1, classifier_dropout=dropout)

#             train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=bs)#,collate_fn=lambda x: model.smart_batching_collate(x))
#             #val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=bs,collate_fn=custom_collate_fn)

#             # Configure the training
#             warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # Warm-up steps
#             logger.info(f"Warmup-steps: {warmup_steps}")

#             evaluator = CEBinaryClassificationEvaluator.from_input_examples(train_samples, name='Relevance BERT')

#             # Train the model
#             model.fit(
#                 train_dataloader=train_dataloader,
#                 evaluator=evaluator,
#                 epochs=num_epochs,
#                 evaluation_steps=0,
#                 optimizer_params={'lr': lr},
#                 warmup_steps=warmup_steps,
#                 output_path=f"{model_save_path}_FINAL"
#             )

###TEST MODEL

# Evaluate the final model on the test set
test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=param_grid['batch_size'][0])
evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='Relevance BERT')

# Load the final model for evaluation
final_model = CrossEncoder(f"model2024-11-07_14-39-42_FINAL")

test_evaluation_result = evaluator(final_model)
logger.info(f"Final Evaluation result: {test_evaluation_result}")

# Calculate MCC
y_true = [example.label for example in test_samples]
y_pred = [int(score > 0.5) for score in final_model.predict(evaluator.sentence_pairs)]
test_mcc = calculate_mcc(y_true, y_pred)
logger.info(f"Final model test MCC: {test_mcc}")

