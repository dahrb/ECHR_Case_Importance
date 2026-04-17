"""
Change log:

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
            importance_val = matched_rows.iloc[0].get('importance')
            # Check if importance_val is valid
            if pd.isna(importance_val) or importance_val not in [1, 2, 3, 4]:
                print(f"Warning: Invalid or missing 'importance' value for appno {appno}.")
                continue  # Skip this appno without updating the dictionary
            importance_labels[itemid] = [appno, json_file, importance_val]
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


def run_get_importance(overlap_directory, output_csv):
    """Extract importance labels from overlap JSON files and save to CSV.

    Args:
        overlap_directory: path to directory with pruned_*_meta.json files.
        output_csv: path where importance_labels.csv will be written.
    """
    importance_labels = {}
    pruned_file = 'pruned_COMMUNICATEDCASES_meta.json'
    pruned_file_path = os.path.join(overlap_directory, pruned_file)
    applications = load_and_extract_ids(pruned_file_path)

    # List of JSON files in the specific order
    json_files = [
        "pruned_GRANDCHAMBER_meta.json",
        "pruned_CHAMBER_meta.json",
        "pruned_COMMITTEE_meta.json",
        "pruned_DECGRANDCHAMBER_meta.json",
        "pruned_ADMISSIBILITY_meta.json",
        "pruned_ADMISSIBILITYCOM_meta.json"
    ]

    for json_file in json_files:
        print(f"JSON file: {json_file}")
        file_path = os.path.join(overlap_directory, json_file)
        if os.path.exists(file_path):
            extract_importance(json_file, file_path, applications, importance_labels)
        else:
            print(f"Warning: File {json_file} does not exist in the directory.")

    data_items = {'itemid': [], 'appno': [], 'source_file': [], 'importance': []}
    for itemid, values in importance_labels.items():
        data_items['itemid'].append(itemid)
        data_items['appno'].append(values[0])
        data_items['source_file'].append(values[1])
        data_items['importance'].append(values[2])

    df = pd.DataFrame(data_items)
    df.to_csv(output_csv, index=False)
    print(f"{output_csv} has been created successfully.")


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--article_dir', dest='article_dir', default=None,
                      help='Path to article directory (e.g. articles/art_3). '
                           'Reads overlap files from <article_dir>/overlap_cases/ and '
                           'writes importance_labels.csv to <article_dir>/. '
                           'If omitted, reads from overlap_cases/ and writes to the current directory.')
    (options, args) = parser.parse_args()

    if options.article_dir:
        overlap_directory = os.path.join(options.article_dir, 'overlap_cases')
        output_csv = os.path.join(options.article_dir, 'importance_labels.csv')
    else:
        overlap_directory = 'overlap_cases'
        output_csv = 'importance_labels.csv'

    run_get_importance(overlap_directory, output_csv)
