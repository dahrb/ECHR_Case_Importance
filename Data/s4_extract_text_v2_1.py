
"""
Version history
v2_1 = adjusts def get_text_tag to remove redundant code and skip tags within a table
  of contents.
v2_0 = extends code to allow for JUDGMENT and DECISION text extraction in addition to
  the COMMUNICATION phase extraction.
v1_2 = corrects several flaws in previous code: (1) Extracts all text content from html nested
  structure, rather than the immediate children; (2) Ensures Appendix is not included in the
  subject matter, where questions to the parties are omitted; (3) Ensures footnotes are not
  included; (4) Includes text formatted in list html.
v1_1 = properly segments into the subject matter and the questions, and structures the txt
file output such that each paragraph is rendered on a new line.
v1_0 = functional code that extracts the text prior to the questions and the questions text.
"""

import copy
import json
import os
import sys
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

if len(sys.argv) != 2:
    raise ValueError("Usage: python s4_extract_text_v2_1.py <article_number>  e.g. 3, 6, 8")
ARTICLE = sys.argv[1]


"""
def check_relevant examines the case to ensure that THE FACTS section is present,
and extracts the relevant passage from the html expression of the case.
"""
def check_passage(case_html, identity, start_strings, end_strings):

    # Check for table of contents and skip any tags within
    contents_strings = ["Table of Contents"]
    contents_check = False
    if get_text_tag(case_html, contents_strings, identity, contents_check):
        contents_check = True
    
    start_tag = get_text_tag(case_html, start_strings, identity, contents_check)
    if not start_tag:
        return ""   
    end_tag = get_text_tag(case_html, end_strings, identity, contents_check)
      
    html_text = ""
    current_tag = start_tag
    #print("start_tag:", start_tag)
    """
    if end_tag:
        print("end_tag:", end_tag.text)
    """
    while current_tag:
        # If end_tag is not None, check if current_tag is end_tag or contains end_string
        if end_tag:
            if current_tag == end_tag or (current_tag.text and end_tag.text in current_tag.text):
                break  # Stop before adding end_tag content
        # If end_tag is None, simply continue until there are no more tags
        html_text += str(current_tag)
        current_tag = current_tag.next_element

    # Convert HTML string to text, ensuring correct handling of <span> elements with &nbsp;
    soup = BeautifulSoup(html_text, 'html.parser')
    paragraphs = soup.find_all(['p', 'ol'])

    # Generate the processed text for all paragraphs
    processed_paragraphs_text = "\n".join(process_paragraph(paragraph) for paragraph in paragraphs)

    return processed_paragraphs_text


def extract_json_ids(chamber_type, file_path=None):
    if file_path is None:
        file_path = f"article{ARTICLE}_cases.json"

    # Open and read the JSON file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
            try:
                # Load the JSON object from the file
                json_obj = json.load(file)
                # Check if the specified chamber_type is a key in the JSON object
                if chamber_type in json_obj:
                    itemid_list = [entry for entry in json_obj[chamber_type]]

            except json.JSONDecodeError:
                print("Error in extract_json_ids decoding JSON from line")
            except KeyError:
                print("Key error in JSON data using extract_json_ids; necessary key may be missing")

    # Optionally return the dictionary or process it further
    return itemid_list


"""
def get_text_tag looks for the string in the html using regular expressions
to identify the appropriate tag or raise an error if it is not found.
"""
def get_text_tag(case_html, target_strings, identity, contents_check):

    for target_string in target_strings:
        # Compile a case-insensitive regex pattern for exact match
        # Use anchors (^ and $) to match the start and end of the string, ensuring exact match
        text_pattern = re.compile(r'^' + re.escape(target_string) + r'$', re.IGNORECASE)
    
        # Find all tags, then filter them manually
        all_tags = case_html.find_all(string=text_pattern)  # True finds all tags

        for tag in all_tags:
            # Check if any direct string child of the tag is an exact match
            for string in tag.strings:
                if text_pattern.match(string):
                    if tag.parent and tag.parent.name == 'span':
                        if contents_check == False:
                            #print("tag.parent:", tag.parent)
                            return tag.parent  # Return the span tag itself if it's the parent of an exact match  
                        else:
                            contents_check = False     
            
    return None
    

def preprocess_html(case_html):
    # Remove all <table> elements from the HTML to prevent their content from being processed
    for table in case_html.find_all('table'):
        table.decompose()


def process_paragraph(paragraph):

    def extract_text(element):
        text = ""
        # Iterate through all child elements recursively
        for child in element:
            # Check for the footnote reference class and stop further text extraction
            if child.name == 'a' and 'ftnref' in child.get('href', ''):
                return text  # Stop processing further elements   
            if isinstance(child, NavigableString):
                text += str(child)
            elif child.name == 'span':
                if child.get_text(strip=True) == '\xa0':  # Handle non-breaking space
                    text += ' '
                else:
                    text += extract_text(child)
            else:
                text += extract_text(child)
        return text  
        
    # Clean up excessive spaces and strip the string of leading/trailing whitespace
    cleaned_text = ' '.join(extract_text(paragraph).split())
    return cleaned_text.strip()
          
    
"""
def scrapecases is the main function in this script. Takes as input parameters: the ECHR article, the desired 
case outcome, and the maximum number of cases to be scraped. Outputs two csv files, one for violation cases 
and the other for nonviolation cases. The output files have four columns containing information on the cases: 
id, date, importance (defined by the ECtHR), and the raw html text.
"""
def scrapecases(itemid_list, start_headers, end_headers, doc_type, passage_type):
    
    missing_list = []

    # Comm-phase text is always stored under corpora/communication_phase/
    # Judgment text is stored under corpora/article{ARTICLE}/{doc_type}/ 
    if doc_type == "COMMUNICATEDCASES":
        base_dir = os.path.join('corpora', 'communication_phase', passage_type)
    else:
        base_dir = os.path.join('corpora', f'article{ARTICLE}', doc_type, passage_type)
    
    for case_item in tqdm(itemid_list): 
        # Extract case html text from HUDOC.
        identity = case_item[1]
        url = "https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id=" + identity
        page = requests.get(url)
        case_html = BeautifulSoup(page.content, 'html.parser')       
        # Extract passage with further preprocessing require if passage is close of document to ensure tables are omitted
        if passage_type == 'questions':      
            preprocess_html(case_html)
        case_passage = check_passage(case_html, identity, start_headers, end_headers)       
        missing_list = write_case(base_dir, case_item[0], identity, case_passage, missing_list)
    
    # Write itemids corresponding to missing elements
    write_missing(f"{doc_type}_{passage_type}_missing", missing_list)    
          
    return    
"""
def write_case saves the desired passage from the given case to the appropriate directory
"""
def write_case(base_dir, doc_date, identity, passage_txt, missing_list):  
    if passage_txt == "":
        missing_list.append(identity)
        return missing_list
    # Save HTML content to file
    save_dir = base_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"{doc_date}_{identity}.txt"), 'w', encoding='utf-8') as file:
        file.write(passage_txt)
    return missing_list


def write_missing(missing_list_type, missing_list):
    if missing_list:
        with open(missing_list_type + ".txt", 'w') as file:
            for itemid in missing_list:
                file.write(itemid + '\n') 
    return


# Note that there are many potential headers due to significant inconsistency in the written structure of the cases
def main(doc_type):

    print(f"Processing {doc_type} cases")

    if doc_type == "COMMUNICATEDCASES":
        # Load the Communicated case CSV file into a DataFrame
        df = pd.read_csv("important_labels.csv")
        # Create a list where each element is a list with doc_date and itemid
        itemid_list = [[date, itemid] for date, itemid in zip(df['doc_date'], df['itemid'])]
        # Relevant header strings that separate the passages
        subject_headers = ["ADDITIONAL FACTS", "ADDITIONAL STATEMENT OF FACTS", "STATEMENR OF FACTS", "STATEMENT OF FACTS", "SUBJECT MATTER OF THE CASE", "SUBJECT MATTER OF THE CASES", "THE CIRCUMSTANCES OF THE CASE", "THE CIRCUMSTANCES OF THE CASES", "UPDATED SUBJECT MATTER OF THE CASE", "UPDATED SUBJECT MATTER OF THE CASES"]
        question_headers = ["ADDITIONAL QUESTION", "ADDITIONAL QUESTION TO THE PARTIES", "ADDITIONAL QUESTIONS", "ADDITIONAL QUESTIONS TO THE PARTIES", "COMMON QUESTION", "COMMON QUESTIONS", "COMMON QUESTIONS TO THE PARTIES", "FACTUAL QUESTIONS TO THE ITALIAN GOVERNMENT", "QUESIONS TO THE PARTIES", "QUESTION", "QUESTION TO THE GOVERNMENT", "QUESTION TO THE NETHERLANDS GOVERNMENT", "QUESTION TO THE PARTIES", "QUESTION TO THE PARTIES:", "QUESTION TO THE PARTIES AND REQUEST", "QUESTION TO THE PARTIES AND REQUEST FOR DOCUMENTS", "QUESTION TO THE PARTIES AND REQUESTS", "QUESTIONS", "QUESTIONS AND REQUESTS TO THE PARTIES", "QUESTIONS APPLICABLE TO ALL THREE APPLICATIONS", "QUESTIONS O THE PARTIES" "QUESTIONS TO THE GOVERNMENT", "QUESTIONS TO THE GOVERNMENT OF GREECE", "QUESTIONS TO THE ITALIAN GOVERNMENT", "QUESTIONS TO THE NETHERLANDS GOVERNMENT", "QUESTIONS TO THE PARTIES", "QUESTIONS TO THE PARTIES AND INFORMATION REQUESTED", "QUESTIONS TO THE PARTIES AND REQUESTS", "QUESTIONS TO THE PARTIES AND REQUEST FOR DOCUMENTS", "QUESTIONS TO THE PARTIES AND REQUEST FOR INFORMATION", "QUESTIONS TO THE PARTIES AS REGARDS ALL THE APPLICATIONS", "QUESTIONSTO THE PARTIES", "REQUEST FOR FACTUAL INFORMATION AND QUESTION TO THE PARTIES", "REQUEST FOR FACTUAL INFORMATION AND QUESTIONS TO THE PARTIES"]
        terminal_headers = ["APPENDIX", "QUESTIONS AUX PARTIES"]
        # Running scrapecases to obtain relevant passages
        scrapecases(itemid_list, subject_headers, question_headers+terminal_headers, doc_type, "subject_matter")
        scrapecases(itemid_list, question_headers, terminal_headers, doc_type, "questions")
    
    else:
        itemid_list = extract_json_ids(doc_type)
        # Relevant header strings that separate the passages
        facts_headers = ["SUBJECT MATTER OF THE CASE", "THE FACTS", "AS TO THE FACTS", "FACTS AND PROCEDURE", "PROCEDURE AND FACTS", "THE FACTS AND PROCEDURE", "PROCEDURE", "THE PROCEDURE", "THE FACTS, RELEVANT DOMESTIC LAW AND PRACTICE AND THE APPLICANT’S COMPLAINTS"]
        law_headers = ["THE LAW", "AS TO THE LAW", "LAW", "COMPLAINTS AND THE LAW", "HE LAW", "THE COURT’S ASSESSMENT", "FOR THESE REASONS, THE COURT, UNANIMOUSLY,"]
        terminal_headers = ["APPENDIX", "ANNEX"]
        # Running scrapecases to obtain relevant passages
        #scrapecases(itemid_list, facts_headers, law_headers+terminal_headers, doc_type, "fact_section")
        scrapecases(itemid_list, law_headers, terminal_headers, doc_type, "law_section")

    return


"""
with open('missing_subject.txt', 'r') as file:
    missing_subject = [line.strip() for line in file]
with open('missing_questions.txt', 'r') as file:
    missing_questions = [line.strip() for line in file]
itemid_list = missing_subject + missing_questions
"""

main("COMMUNICATEDCASES")

main("ADMISSIBILITYCOM")
main("ADMISSIBILITY")
main("DECGRANDCHAMBER")

main("COMMITTEE")
main("CHAMBER")
main("GRANDCHAMBER")
