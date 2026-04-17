import json

# Load the JSONL data
input_file = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/FINE-TUNE/test_data.jsonl'
output_file = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/FINE-TUNE/llama_test_data.jsonl'

formatted_data = []

with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        messages = data['messages']
        
        # Extract the user prompt and the expected completion
        for message in messages:
            if message['role'] == 'user':
                prompt = message['content']
            elif message['role'] == 'assistant':
                completion = message['content']
        
        # Format the data
        formatted_entry = {
            "prompt": prompt,
            "completion": completion
        }
        formatted_data.append(formatted_entry)

# Save the formatted data
with open(output_file, 'w') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + '\n')