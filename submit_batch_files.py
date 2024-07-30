"""
Version history
v1_0 = implements ability to send files to api and download the results
"""

from openai import OpenAI
from API_key import openai_key
import os

client = OpenAI(api_key=openai_key)

def send_to_api(filepath,experiment_desciption):

    '''
    Sends files to OpenAI API for batch processing

    Experiment_description naming conventions =
        'Experiment {number} {type}:'
    '''
    
    for i in os.listdir(f'{filepath}'):
        if i.endswith('.jsonl'):
            batch_input_file = client.files.create(
                file=open(filepath+f"/{i}", "rb"),
                purpose="batch"
                )

            batch_input_file_id = batch_input_file.id

            client.batches.create(
            input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": f"{experiment_desciption} {i}"
                })
            
def download_batch_files(experiment_name,filepath):

    '''
    Downloads files from OpenAI API after batch processing
    '''

    for i in client.batches.list():
        
        if i.metadata['description'].split(':')[0] == experiment_name:
            #print(i.metadata['description'].split(':')[0])
            
            file_id = i.output_file_id
            #print(file_id)
            batch_output_file = client.files.content(file_id).content

            with open(f'{filepath}/{i.metadata["description"].split(":")[1]}', "wb") as f:
                f.write(batch_output_file)
    
if __name__ == '__main__':
    #send all experiments in the folder to the api
    send_to_api(filepath='./Batches/experiment_2_CoT',experiment_desciption='Experiment 2_CoT: ')

    #uncomment and download the files after processing - maybe 5-10mins after running send_to_api
    #download_batch_files(experiment_name='Experiment 2_CoT',filepath='./Results/experiment_2_CoT')