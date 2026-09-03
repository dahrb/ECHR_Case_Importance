import pandas as pd
from summarize_cases import prep_prompt, save_file

outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')
outcome_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/outcome_summaries.pkl')

filenames_in_cases_not_in_summaries = outcome_cases[~outcome_cases['File'].isin(outcome_summaries['Filename'])]

filenames_in_cases_not_in_summaries = pd.DataFrame(filenames_in_cases_not_in_summaries)

filenames_in_cases_not_in_summaries = filenames_in_cases_not_in_summaries.drop(columns=['The Law'])

outcome_output = prep_prompt(filenames_in_cases_not_in_summaries,prompt_type='outcome')

save_file(outcome_output,'/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases','outcome_final_cases_3')

