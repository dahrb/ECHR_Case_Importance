"""
Change log:

v1_1 = Added logging the document date to csv output.
v1_0 = Extends from s3_get_importance_v1_1.py to obtain key words in addition to importance labels

* from s3_get_importance_v1_1.py
v1_1 = Adapts dictionary where communicated case itemid is the key and a list of appno, 
source file, and importance is the value.
v1_0 = Returns dictionary where appno is the key and a list of source file and importance
is the value.
"""

import os
import pandas as pd


def extract_importance(json_file, file_path, applications, importance_labels):
    """Extracts importance for matching appnos and updates the labels dictionary."""
    df = pd.read_json(file_path, lines=True)
    # Iterate through each application number remaining in the applications list
    for itemid, appno in list(applications.items()):  # Use list to make a copy since we modify applications inside the loop
        matched_rows = df[df['appno'] == appno]
        if not matched_rows.empty:
            doc_date = matched_rows.iloc[0].get('kpdate').split('T')[0]
            importance_val = matched_rows.iloc[0].get('importance')
            key_words = matched_rows.iloc[0].get('kpthesaurus') # The key words are actually the keys to the dictionary found in key_labels.json
            # Check if importance_val is valid
            if pd.isna(importance_val) or importance_val not in [1, 2, 3, 4]:
                print(f"Warning: Invalid or missing 'importance' value for appno {appno}.")
                continue  # Skip this appno without updating the dictionary
            important_labels[itemid] = [appno, json_file, doc_date, importance_val, key_words]
            del applications[itemid]  # Remove the itemid key-value pair so it is not used again


def load_and_extract_ids(filename):
    """Load a JSON file with lines=True and create a dictionary with 'itemid' as keys and 'appno' as values."""
    try:
        df = pd.read_json(filename, lines=True)
        # Drop any rows where either 'itemid' or 'appno' is NaN to ensure data integrity
        df = df.dropna(subset=['itemid', 'appno'])
        # Create a dictionary from the DataFrame
        id_appno_dict = pd.Series(df['appno'].values, index=df['itemid']).to_dict()
        return id_appno_dict
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return {}


# Directory containing the JSON files
directory = 'overlap_cases'

# Initial empty dictionary and applications list taken from Communicated Cases
important_labels = {}
pruned_file = 'pruned_COMMUNICATEDCASES_meta.json'
pruned_file_path = os.path.join(directory, pruned_file)
applications = (load_and_extract_ids(pruned_file_path))

# List of JSON files in the specific order
json_files = [
    "pruned_GRANDCHAMBER_meta.json",
    "pruned_CHAMBER_meta.json",
    "pruned_COMMITTEE_meta.json",
    "pruned_DECGRANDCHAMBER_meta.json",
    "pruned_ADMISSIBILITY_meta.json",
    "pruned_ADMISSIBILITYCOM_meta.json"
]

# Iterate through each specified file in the given order
for json_file in json_files:
    print(f"JSON file: {json_file}")
    file_path = os.path.join(directory, json_file)
    if os.path.exists(file_path):  # Check if file exists to prevent errors
        extract_importance(json_file, file_path, applications, important_labels)
    else:
        print(f"Warning: File {json_file} does not exist in the directory.")

# Convert the dictionary into a DataFrame
data_items = {'itemid': [],'appno': [], 'source_file': [], 'doc_date': [], 'importance': [], 'key_words_keys': []}
for itemid, values in important_labels.items():
    data_items['itemid'].append(itemid)
    data_items['appno'].append(values[0])
    data_items['source_file'].append(values[1]) 
    data_items['doc_date'].append(values[2])  
    data_items['importance'].append(values[3])  
    data_items['key_words_keys'].append(values[4])

df = pd.DataFrame(data_items)

# Save the DataFrame to a CSV file
df.to_csv('important_labels.csv', index=False)

print("important_labels.csv has been created successfully.")
