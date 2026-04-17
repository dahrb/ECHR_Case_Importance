import os
import json
import pandas as pd

def load_json(filename):
    """ Helper function to load a JSON file saved with lines=True into a DataFrame. """
    return pd.read_json(filename, lines=True)

def save_json(df, filename):
    """ Helper function to save a DataFrame to a JSON file. """
    df.to_json(filename, orient='records', lines=True)


def run_overlap(raw_metadata_dir, overlap_directory):
    """Compute overlapping cases between COMMUNICATEDCASES and judgment/decision metadata.

    Args:
        raw_metadata_dir: directory containing the *_meta.json files produced by s1.
        overlap_directory: directory where pruned overlap files will be written.
    """
    os.makedirs(overlap_directory, exist_ok=True)

    comm_path = os.path.join(raw_metadata_dir, 'COMMUNICATEDCASES_meta.json')
    # Load the communicated cases
    comm_cases = pd.read_json(comm_path, lines=True)
    comm_appnos = set(comm_cases['appno'].dropna())  # Set of appno values from communicated cases

    # To store appnos found in other files for pruning COMMUNICATEDCASES_meta later
    other_appnos = set()

    # Iterate over files in the raw_metadata directory
    for filename in os.listdir(raw_metadata_dir):
        if filename.endswith('.json') and filename != 'COMMUNICATEDCASES_meta.json':
            filepath = os.path.join(raw_metadata_dir, filename)
            # Load the current JSON file
            df = pd.read_json(filepath, lines=True)

            # Filter rows where appno matches any in COMMUNICATEDCASES_meta.json
            overlap_df = df[df['appno'].isin(comm_appnos)]

            # Update the set of appnos found in other files
            other_appnos.update(df['appno'].dropna())

            # Save the overlap file with pruned_ prefix (convention expected by s3)
            overlap_filename = os.path.join(overlap_directory, 'pruned_' + filename)
            save_json(overlap_df, overlap_filename)

    # Prune COMMUNICATEDCASES_meta.json based on appnos found in other JSON files
    overlap_comm_df = comm_cases[comm_cases['appno'].isin(other_appnos)]
    overlap_comm_filename = os.path.join(overlap_directory, 'pruned_COMMUNICATEDCASES_meta.json')
    save_json(overlap_comm_df, overlap_comm_filename)

    print(f"Pruning complete. Overlap files are saved in '{overlap_directory}'.")


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--article_dir', dest='article_dir', default=None,
                      help='Path to article directory (e.g. articles/art_3). '
                           'Reads raw metadata from <article_dir>/raw_metadata/ and '
                           'writes overlap files to <article_dir>/overlap_cases/. '
                           'If omitted, reads from the current directory and writes to overlap_cases/.')
    (options, args) = parser.parse_args()

    if options.article_dir:
        raw_metadata_dir = os.path.join(options.article_dir, 'raw_metadata')
        overlap_directory = os.path.join(options.article_dir, 'overlap_cases')
    else:
        raw_metadata_dir = '.'
        overlap_directory = 'overlap_cases'

    run_overlap(raw_metadata_dir, overlap_directory)