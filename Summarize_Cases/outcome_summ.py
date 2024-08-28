import pandas as pd
from summarize_cases import prep_prompt, save_file

outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')

outcome_output = prep_prompt(outcome_cases,prompt_type='outcome')

save_file(outcome_output,'/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases','outcome')
