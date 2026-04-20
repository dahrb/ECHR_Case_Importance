import os
import json
import pandas as pd

def load_json(filename):
    """ Helper function to load a JSON file saved with lines=True into a DataFrame. """
    return pd.read_json(filename, lines=True)

def save_json(df, filename):
    """ Helper function to save a DataFrame to a JSON file. """
    df.to_json(filename, orient='records', lines=True)

# Path to the directory containing JSON files
overlap_directory = 'overlap_cases'
os.makedirs(overlap_directory, exist_ok=True)

# Load the communicated cases
comm_cases = pd.read_json('COMMUNICATEDCASES_meta.json', lines=True)
comm_appnos = set(comm_cases['appno'].dropna())  # Set of appno values from communicated cases

# To store appnos found in other files for pruning COMMUNICATEDCASES_meta later
other_appnos = set()

# Iterate over files in the directory
for filename in os.listdir('.'):
    if filename.endswith('.json') and filename != 'COMMUNICATEDCASES_meta.json':
        # Load the current JSON file
        df = pd.read_json(filename, lines=True)

        # Filter rows where appno matches any in COMMUNICATEDCASES_meta.json
        overlap_df = df[df['appno'].isin(comm_appnos)]

        # Update the set of appnos found in other files
        other_appnos.update(df['appno'].dropna())

        # Save the overlap file
        overlap_filename = os.path.join(overlap_directory, filename)
        save_json(overlap_df, overlap_filename)

# Prune COMMUNICATEDCASES_meta.json based on appnos found in other JSON files
overlap_comm_df = comm_cases[comm_cases['appno'].isin(other_appnos)]
overlap_comm_filename = os.path.join(overlap_directory, 'COMMUNICATEDCASES_meta.json')
save_json(overlap_comm_df, overlap_comm_filename)

print("Pruning complete. overlap files are saved in the 'overlap_cases' directory.")