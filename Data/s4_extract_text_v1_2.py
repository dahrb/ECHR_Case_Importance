
"""
Version history
v1_2 = corrects several flaws in previous code: (1) Extracts all text content from html nested
  structure, rather than the immediate children; (2) Ensures Appendix is not included in the
  subject matter, where questions to the parties are omitted; (3) Ensures footnotes are not
  included; (4) Includes text formatted in list html.
v1_1 = properly segments into the subject matter and the questions, and structures the txt
file output such that each paragraph is rendered on a new line.
v1_0 = functional code that extracts the text prior to the questions and the questions text.
"""

import copy
import os
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm


"""
def check_relevant examines the case to ensure that THE FACTS section is present,
and extracts the relevant passage from the html expression of the case.
"""
def check_passage(case_html, identity, start_strings, end_strings):
    
    start_tag = get_text_tag(case_html, start_strings, identity)
    if not start_tag:
        return ""   
    end_tag = get_text_tag(case_html, end_strings, identity)
      
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


"""
def get_text_tag looks for the string in the html using regular expressions
to identify the appropriate tag or raise an error if it is not found.
"""
def get_text_tag(case_html, target_strings, identity):
    for target_string in target_strings:
        # Compile a case-insensitive regex pattern for exact match
        # Use anchors (^ and $) to match the start and end of the string, ensuring exact match
        text_pattern = re.compile(r'^' + re.escape(target_string) + r'$', re.IGNORECASE)
    
        # Find all tags, then filter them manually
        all_tags = case_html.find_all(string=text_pattern)  # True finds all tags
        exact_match_tags = []
    
        for tag in all_tags:
            # Check if any direct string child of the tag is an exact match
            for string in tag.strings:
                if text_pattern.match(string):
                    exact_match_tags.append(tag)
                    break  # Break if you only need the first exact match within a single tag
    
        # Filter based on your specific condition (e.g., parent is a 'span' tag)
        for tag in exact_match_tags:
            if tag.parent and tag.parent.name == 'span':
                #print("tag.parent:", tag.parent)
                return tag.parent  # Return the span tag itself if it's the parent of an exact match
            
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
def scrapecases(itemid_list, output_dir=None):
    
    missing_subject = []
    missing_questions = []

    for identity in tqdm(itemid_list): 
        # Extract case html text from HUDOC.
        url = "https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id=" + identity
        page = requests.get(url)
        case_html = BeautifulSoup(page.content, 'html.parser')
        
        # Strings that separate subject matter of the case (inconsistently
        # labelled) from the questions to the parties.
        subject_headers = ["ADDITIONAL FACTS", "ADDITIONAL STATEMENT OF FACTS", "STATEMENR OF FACTS", "STATEMENT OF FACTS", "SUBJECT MATTER OF THE CASE", "SUBJECT MATTER OF THE CASES", "THE CIRCUMSTANCES OF THE CASE", "THE CIRCUMSTANCES OF THE CASES", "UPDATED SUBJECT MATTER OF THE CASE", "UPDATED SUBJECT MATTER OF THE CASES"]
        question_headers = ["ADDITIONAL QUESTION", "ADDITIONAL QUESTION TO THE PARTIES", "ADDITIONAL QUESTIONS", "ADDITIONAL QUESTIONS TO THE PARTIES", "COMMON QUESTION", "COMMON QUESTIONS", "COMMON QUESTIONS TO THE PARTIES", "FACTUAL QUESTIONS TO THE ITALIAN GOVERNMENT", "QUESIONS TO THE PARTIES", "QUESTION", "QUESTION TO THE GOVERNMENT", "QUESTION TO THE NETHERLANDS GOVERNMENT", "QUESTION TO THE PARTIES", "QUESTION TO THE PARTIES:", "QUESTION TO THE PARTIES AND REQUEST", "QUESTION TO THE PARTIES AND REQUEST FOR DOCUMENTS", "QUESTION TO THE PARTIES AND REQUESTS", "QUESTIONS", "QUESTIONS AND REQUESTS TO THE PARTIES", "QUESTIONS APPLICABLE TO ALL THREE APPLICATIONS", "QUESTIONS O THE PARTIES" "QUESTIONS TO THE GOVERNMENT", "QUESTIONS TO THE GOVERNMENT OF GREECE", "QUESTIONS TO THE ITALIAN GOVERNMENT", "QUESTIONS TO THE NETHERLANDS GOVERNMENT", "QUESTIONS TO THE PARTIES", "QUESTIONS TO THE PARTIES AND INFORMATION REQUESTED", "QUESTIONS TO THE PARTIES AND REQUESTS", "QUESTIONS TO THE PARTIES AND REQUEST FOR DOCUMENTS", "QUESTIONS TO THE PARTIES AND REQUEST FOR INFORMATION", "QUESTIONS TO THE PARTIES AS REGARDS ALL THE APPLICATIONS", "QUESTIONSTO THE PARTIES", "REQUEST FOR FACTUAL INFORMATION AND QUESTION TO THE PARTIES", "REQUEST FOR FACTUAL INFORMATION AND QUESTIONS TO THE PARTIES"]
        terminal_headers = ["APPENDIX", "QUESTIONS AUX PARTIES"]
        
        # Extract subject matter and questions to the parties
        subject_matter = check_passage(case_html, identity, subject_headers, question_headers+terminal_headers)       
        preprocess_html(case_html)
        questions = check_passage(case_html, identity, question_headers, terminal_headers)
        
        missing_subject = write_case("subject_matter", identity, subject_matter, missing_subject, output_dir)
        missing_questions = write_case("questions", identity, questions, missing_questions, output_dir)  
    
    # Write itemids corresponding to missing elements
    write_missing("subject_missing", missing_subject, output_dir)
    write_missing("questions_missing", missing_questions, output_dir)
                
    return
    
    
"""
def write_case saves the desired passage from the given case to the appropriate directory
"""
def write_case(passage_type, identity, passage_txt, missing_list, output_dir=None):  
    if passage_txt == "":
        missing_list.append(identity)
        return missing_list
    # Save HTML content to file
    if output_dir:
        passage_dir = os.path.join(output_dir, passage_type)
    else:
        passage_dir = passage_type
    os.makedirs(passage_dir, exist_ok=True)
    with open(os.path.join(passage_dir, identity + ".txt"), 'w', encoding='utf-8') as file:
        file.write(passage_txt)
    return missing_list


def write_missing(missing_list_type, missing_list, output_dir=None):
    if missing_list:
        filename = missing_list_type + ".txt"
        if output_dir:
            filename = os.path.join(output_dir, filename)
        with open(filename, 'w') as file:
            for itemid in missing_list:
                file.write(itemid + '\n') 
    return


"""
def scrapecases(article, outcome, limit). Where article variable is an integer indicating the article 
(e.g. 6); outcome variable is a string either "violation" or "nonviolation"; and limit variable is an 
integer that denotes the upper limit for number of scraped cases as determined by the relevant HUDOC query.
"""
if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--article_dir', dest='article_dir', default=None,
                      help='Path to article directory (e.g. articles/art_3). '
                           'Reads importance_labels.csv from <article_dir>/ and '
                           'writes text to <article_dir>/corpora/communication_phase/. '
                           'If omitted, reads importance_labels.csv from the current directory '
                           'and writes subject_matter/ and questions/ into the current directory.')
    (options, args) = parser.parse_args()

    if options.article_dir:
        labels_csv = os.path.join(options.article_dir, 'importance_labels.csv')
        output_dir = os.path.join(options.article_dir, 'corpora', 'communication_phase')
    else:
        labels_csv = 'importance_labels.csv'
        output_dir = None

    import pandas as pd
    df = pd.read_csv(labels_csv)
    itemid_list = df['itemid'].tolist()
    scrapecases(itemid_list, output_dir)