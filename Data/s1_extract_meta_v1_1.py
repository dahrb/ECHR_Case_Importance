"""
This script identifies those communicated cases on HUDOC that correspond to a final judgement,
and extracts the text from the communication phase, segmented into two partitions: The Subject
Matter Of The Case, and The Questions To The Parties. The extracted text is saved in JSON format.

***
Note

fields = ['itemid','applicability','application','appno','article','conclusion','decisiondate','docname',
'documentcollectionid','documentcollectionid2','doctype','doctypebranch','ecli','externalsources','extractedappno',
'importance','introductiondate','isplaceholder','issue','judgementdate','kpdate','kpdateAsText','kpthesaurus',
'languageisocode','meetingnumber','originatingbody','publishedby','Rank','referencedate','reportdate','representedby',
'resolutiondate', 'resolutionnumber','respondent','respondentOrderEng','rulesofcourt','separateopinion','scl',
'sharepointid','typedescription','nonviolation','violation', 'ECHRRanking', 'languagenumber', 'advopidentifier', 
'advopstatus', 'appnoparts', 'sclappnos']
***

Version history
v1_0 = functional code that saves metadata of ECtHR judgment cases and communicated cases.
"""

import os
import pandas as pd
import requests
from time import sleep
from tqdm import tqdm


"""
def case_meta is the main function in this script. Takes as input parameters: the ECHR article, the desired 
case outcome, and the maximum number of cases to be scraped. Outputs two csv files, one for violation cases 
and the other for nonviolation cases. The output files have four columns containing information on the cases: 
id, date, importance (defined by the ECtHR), and the raw html text.
"""
def case_meta(query, output_dir=None):
    
    print(f"Running query: {query}")

    limit = 10000 # Limit is server-based, HUDOC will not return results beyond this hard limit
    start = 0
    length = 500
    all_data = []

    """
    These numbers are set to cover all extant case law, they must be reviewed before the function is
    called in order to ensure they are sufficient.
    """
    if query in ["COMMUNICATEDCASES","ADMISSIBILITYCOM","ADMISSIBILITY","CHAMBER"]:
        kpdate = ' AND ((kpdate>="2019-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate>="2014-05-28T00:00:00.0Z")) AND ((kpdate<"2019-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate>="2009-05-28T00:00:00.0Z")) AND ((kpdate<"2014-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data) 
        kpdate = ' AND ((kpdate>="2004-05-28T00:00:00.0Z")) AND ((kpdate<"2009-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)      
    elif query in ["DECGRANDCHAMBER","COMMITTEE","GRANDCHAMBER"]:
        kpdate = ""
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
    else:
        raise ValueError(f"Unexpected document type: {query}")
    
    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(all_data)
    
    # Save the DataFrame to a JSON file, naming the file based on the query
    json_filename = f"{query}_meta.json"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_filename = os.path.join(output_dir, json_filename)
    df.to_json(json_filename, orient='records', lines=True)
    print('\n', f"Number of case records searched: {len(all_data)}")
    print(f"Data saved to {json_filename}")
    return
    
    
def run_loop(limit, start, length, query, kpdate, all_data):

    print(f"kpdate, if appropriate, is: {kpdate}")

    ## Iterating through the urls until the hard limit is reached.   
    while start < limit:
        url = f"https://hudoc.echr.coe.int/app/query/results?query=contentsitename:ECHR AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) AND ((languageisocode=\"ENG\")) AND ((documentcollectionid=\"{query}\")){kpdate}&select=sharepointid,Rank,ECHRRanking,issue,languagenumber,itemid,article,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,kpdateAsText,documentcollectionid,documentcollectionid2,languageisocode,extractedappno,isplaceholder,doctypebranch,respondent,advopidentifier,advopstatus,nonviolation,violation,ecli,appnoparts,applicability,decisiondate,externalsources,introductiondate,issue,judgementdate,kpthesaurus,meetingnumber,publishedby,referencedate,reportdate,representedby,resolutiondate,resolutionnumber,respondentOrderEng,rulesofcourt,separateopinion,scl,sclappnos&sort=&start={start}&length={length}&rankingModelId=11111111-0000-0000-0000-000000000000"
        
        response = requests.get(url)
        data = response.json()
        
        # Check if the results list is empty and break the loop if it is
        if not data['results']:
            print("No more data to fetch.")
            break
        
        for result in data['results']:
            all_data.append(result['columns'])
        
        start += length
        sleep(1)  # Sleep to prevent overloading the server
    
    print(f"len(all_data): {len(all_data)}")
    return all_data


# Program usage
#case_meta("COMMUNICATEDCASES")

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--article_dir', dest='article_dir', default=None,
                      help='Path to article directory (e.g. articles/art_3). '
                           'Outputs will be written to <article_dir>/raw_metadata/. '
                           'If omitted, files are written to the current directory.')
    (options, args) = parser.parse_args()

    output_dir = None
    if options.article_dir:
        output_dir = os.path.join(options.article_dir, 'raw_metadata')

    case_meta("COMMUNICATEDCASES", output_dir)
    case_meta("ADMISSIBILITYCOM", output_dir)
    case_meta("ADMISSIBILITY", output_dir)
    case_meta("DECGRANDCHAMBER", output_dir)
    case_meta("COMMITTEE", output_dir)
    case_meta("CHAMBER", output_dir)
    case_meta("GRANDCHAMBER", output_dir)
