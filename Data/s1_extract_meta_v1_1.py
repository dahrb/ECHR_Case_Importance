"""
This script identifies those communicated cases on HUDOC that correspond to a final judgement,
and extracts the text from the communication phase, segmented into two partitions: The Subject
Matter Of The Case, and The Questions To The Parties. The extracted text is saved in JSON format.

***
Note

fields = ['advopidentifier','advopstatus','applicability','application','appno','appnoparts','article',
'conclusion','decisiondate','docname','doctype','doctypebranch','documentcollectionid','documentcollectionid2',
'ECHRRanking','ecli','externalsources','extractedappno','importance','introductiondate','isplaceholder','issue',
'itemid','judgementdate','keywords''kpdate','kpdateAsText','kpthesaurus','languageisocode','languagenumber',
'meetingnumber','nonviolation','originatingbody','publishedby','Rank','referencedate','reportdate','representedby',
'resolutiondate','resolutionnumber','respondent','respondentOrderEng','rulesofcourt','scl','sclappnos',
'separateopinion','sharepointid','typedescription','violation']
***

Version history
v1_1 = amended date searching in def case_meta so that all cases are included, not just those from 2004 onwards.
v1_0 = functional code that saves metadata of ECtHR judgment cases and communicated cases.
"""

import os
import pandas as pd
import requests
from time import sleep
from tqdm import tqdm


"""
def case_meta is the main function in this script. Takes as input parameters: the ECHR article, the desired case 
outcome, and the maximum number of cases to be scraped. Outputs a json file for the given query (chamber type) 
that contains all relevant cases with their HUDOC metadata values (as indicated by the fields in the preamble).
"""
def case_meta(query):
    
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
        kpdate = ' AND ((kpdate>="2019-05-28T00:00:00.0Z")) AND ((kpdate<"2024-06-13T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate>="2014-05-28T00:00:00.0Z")) AND ((kpdate<"2019-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate>="2009-05-28T00:00:00.0Z")) AND ((kpdate<"2014-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data) 
        kpdate = ' AND ((kpdate>="2004-05-28T00:00:00.0Z")) AND ((kpdate<"2009-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data) 
        kpdate = ' AND ((kpdate>="1999-05-28T00:00:00.0Z")) AND ((kpdate<"2004-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate>="1994-05-28T00:00:00.0Z")) AND ((kpdate<"1999-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate>="1989-05-28T00:00:00.0Z")) AND ((kpdate<"1994-05-28T00:00:00.0Z"))'
        all_data = run_loop(limit, start, length, query, kpdate, all_data)
        kpdate = ' AND ((kpdate<"1994-05-28T00:00:00.0Z"))'
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
    df.to_json(json_filename, orient='records', lines=True)
    print('\n', f"Number of case records searched: {len(all_data)}")
    print(f"Data saved to {json_filename}")
    return
    
    
def run_loop(limit, start, length, query, kpdate, all_data):

    print(f"kpdate, if appropriate, is: {kpdate}")

    ## Iterating through the urls until the hard limit is reached.   
    while start < limit:
    
        url = f"https://hudoc.echr.coe.int/app/query/results?query=contentsitename:ECHR AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) AND ((languageisocode=\"ENG\")) AND ((documentcollectionid=\"{query}\")){kpdate}&select=sharepointid,advopidentifier,advopstatus,applicability,application,appno,appnoparts,article,conclusion,decisiondate,docname,doctype,doctypebranch,documentcollectionid,documentcollectionid2,ECHRRanking,ecli,externalsources,extractedappno,importance,introductiondate,isplaceholder,issue,itemid,judgementdate,keyword,kpdate,kpdateAsText,kpthesaurus,languageisocode,languagenumber,meetingnumber,nonviolation,originatingbody,publishedby,Rank,referencedate,reportdate,representedby,resolutiondate,resolutionnumber,respondent,respondentOrderEng,rulesofcourt,scl,sclappnos,separateopinion,typedescription,violation&sort=&start={start}&length={length}&rankingModelId=11111111-0000-0000-0000-000000000000"
        
        response = requests.get(url)
        data = response.json()
        
        # Check if the results list is empty and break the loop if it is
        if not data['results']:
            print("No more data to fetch.")
            break
        
        for result in data['results']:
            all_data.append(result['columns'])
        
        start += length
        if start >= limit:
            raise ValueError(f"Too many cases for kpdate: {kpdate}. Need to split time interval.")
        sleep(1)  # Sleep to prevent overloading the server
    
    print(f"len(all_data): {len(all_data)}")
    return all_data


# Program usage
case_meta("COMMUNICATEDCASES")

case_meta("ADMISSIBILITYCOM")
case_meta("ADMISSIBILITY")
case_meta("DECGRANDCHAMBER")

case_meta("COMMITTEE")
case_meta("CHAMBER")
case_meta("GRANDCHAMBER")
