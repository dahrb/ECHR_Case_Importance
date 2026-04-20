import json
import os

directory = 'pruned_cases'

for filename in os.listdir('pruned_cases'):
    if filename.endswith('.json'):
        # Initialize a count variable
        number_of_entries = 0
        
        # Read the JSON file line by line
        with open(os.path.join(directory, filename), 'r') as file:
            for line in file:
                # Each line contains a separate JSON object
                json_object = json.loads(line)  # Parse each line as JSON
                number_of_entries += 1  # Increment the count for each JSON object
        
        print(f'The number of entries in the JSON file {filename} is: {number_of_entries}', '\n')
