"""
This script identifies produces A dictionary where the HUDOC kpthesaurus field 
provides keys, and the key words from HUDOC are the values.

Version history
v1_0 = functional code that outputs the dictionary as a json file using the html 
file with the correct key words taxonomy as at 17th July 2024.
"""

import json
from bs4 import BeautifulSoup

# Load the HTML file
with open('key_labels_taxonomy.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

# Initialize BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Initialize the dictionary to store key-value pairs
thesaurus_dict = {}

# Find all <li> elements with class "branch"
for li in soup.find_all('li', class_='branch'):
    # Extract the key from the id attribute of <li>
    key = li.get('id')
    
    # Find the <h4 class="branchname"> element and get its contents
    h4 = li.find('h4', class_='branchname')
    if h4:
        # Extract the text, excluding the <span class="branchcount">
        text = ''.join(h4.find_all(text=True, recursive=False)).strip()
        thesaurus_dict[key] = text

# Save the dictionary to a JSON file
with open('key_labels.json', 'w', encoding='utf-8') as json_file:
    json.dump(thesaurus_dict, json_file, ensure_ascii=False, indent=4)

print("Data saved to extracted_data.json")