import pandas as pd
from summarize_cases import prep_prompt, save_file

comm_cases_train = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/train.pkl')
comm_cases_test = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')

comm_output_train = prep_prompt(comm_cases_train,prompt_type='comm')
comm_output_test = prep_prompt(comm_cases_test,prompt_type='comm')

save_file(comm_output_train,'/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases','comm_train')
save_file(comm_output_test,'/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases','comm_test')
