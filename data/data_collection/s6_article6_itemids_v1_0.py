"""
This script identifies those communicated cases on HUDOC that correspond to a final judgement,
and extracts the text from the communication phase, segmented into two partitions: The Subject
Matter Of The Case, and The Questions To The Parties. The extracted text is saved in JSON format.

Version history
v1_1 = parameterised to accept article number as a CLI argument.
v1_0 = extracts itemids from json files for Article 3 cases.

Usage:
    python s6_article6_itemids_v1_0.py <article_number>
    e.g. python s6_article6_itemids_v1_0.py 3
         python s6_article6_itemids_v1_0.py 6
         python s6_article6_itemids_v1_0.py 8
"""

import json
import os
import sys

if len(sys.argv) != 2:
    raise ValueError("Usage: python s6_article6_itemids_v1_0.py <article_number>")
ARTICLE = sys.argv[1]


def get_itemids(itemid_dict, chamber_type):

    itemid_dict[chamber_type] = []
    
    # Open and read the JSON file line by line
    with open(os.path.join("raw_case_metadata", f"{chamber_type}_meta.json"), 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Load the JSON object from line
                json_obj = json.loads(line)
                
                # Split the 'article' field into a list by ';'
                article_numbers = json_obj['article'].split(';')
                outcome_date = json_obj['kpdate'].split('T')[0]
                
                # Check if the target article is exactly in the list of article numbers
                if ARTICLE in article_numbers:
                    # Add the 'itemid' to the list
                    itemid_dict[chamber_type].append([outcome_date, json_obj['itemid']])
            except json.JSONDecodeError:
                print("Error decoding JSON from line")
            except KeyError:
                print("Key error in JSON data; necessary key may be missing")
    
    return itemid_dict


itemid_dict = {}
get_itemids(itemid_dict, 'GRANDCHAMBER')
get_itemids(itemid_dict, 'CHAMBER')
get_itemids(itemid_dict, 'COMMITTEE')
get_itemids(itemid_dict, 'DECGRANDCHAMBER')
get_itemids(itemid_dict, 'ADMISSIBILITY')
get_itemids(itemid_dict, 'ADMISSIBILITYCOM')

# Save the dictionary to a JSON file
output_file = f'article{ARTICLE}_cases.json'
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(itemid_dict, json_file, ensure_ascii=False, indent=4)
print(f"Saved itemids for Article {ARTICLE} to {output_file}")