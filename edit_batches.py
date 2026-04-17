import json
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

filepath = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/BASELINES_NEW/prediction_baselines_2_True_1_13_random_state_42.jsonl'

data = []
with open(filepath, 'r') as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)

#df['response'] = df['response'].apply(lambda x: x['choices'][0]['message']['content'])

#Give a separate output for each importance category: 3, 2, 1, key_case.

JSON_SCHEMAS = [{"Case Importance":"string (select one of: key_case, 1, 2, 3)",'Confidence in Importance Ascription': 'int 1-100', "Reasoning":"string (give your reason for the importance)" }
                ]

for index, row in df.iterrows():
    row['body']['messages'][0]['content'] = row['body']['messages'][0]['content'].replace(
        "{'Case Importance': 'string (select one of: key_case, 1, 2, 3)', 'Reasoning': 'string (give your reason for the importance)'}", 
        str(JSON_SCHEMAS[0])
    )
    row['body']['messages'][0]['content'] += "\nGive a separate JSON structured output for each importance category: 3, 2, 1, key_case."

    row['body']['max_tokens'] = 2048
    row['body']['response_format'] = {"type": "text"}

output_filepath = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/BASELINES_NEW/ITER_ZERO_SHOT_BASELINE.jsonl'

with open(output_filepath, 'w') as file:
    counter = 0
    for record in df.to_dict(orient='records'):
        file.write(json.dumps(record) + '\n')