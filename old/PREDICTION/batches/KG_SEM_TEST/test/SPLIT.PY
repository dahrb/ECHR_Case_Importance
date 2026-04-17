import pandas as pd
import json


filepath = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/KG_SEM_TEST/test/predictions_3_OPENAI-2048-COSINE.jsonl'

data = pd.read_json(filepath,lines=True)

# Split the data into two halves
half = len(data) // 2
data_first_half = data.iloc[:half]
data_second_half = data.iloc[half:]

# Write each half to a different .jsonl file
data_first_half.to_json('/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/KG_SEM_TEST/ANNOY/predictions_3_OPENAI-2048-COSINE_first_half.jsonl', orient='records', lines=True)
data_second_half.to_json('/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/KG_SEM_TEST/ANNOY/predictions_3_OPENAI-2048-COSINE_second_half.jsonl', orient='records', lines=True)
