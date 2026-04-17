import pandas as pd
from summarize_cases import prep_prompt, save_file

#comm_cases_train = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/train.pkl')
comm_cases_test = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')

test_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/comm_test_summaries.pkl')

# Filter out rows in comm_cases_test whose filenames are in test_summaries
filtered_cases_test = comm_cases_test[~comm_cases_test['Filename'].isin(test_summaries['Filename'])]

#comm_output_train = prep_prompt(comm_cases_train,prompt_type='comm')
comm_output_test = prep_prompt(filtered_cases_test,prompt_type='comm')

### edit the files to have only new cases to summarise!

#save_file(comm_output_train,'/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases','comm_train')
save_file(comm_output_test,'/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases','comm_test_pt_2')
