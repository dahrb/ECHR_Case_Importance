import pandas as pd
import json
import os
from sklearn.metrics import precision_score, recall_score


def data_process(data):
    data = data[['Filename','keywords_art_3']]
    return data

def combine_results(filepath,keyword = 'Case Importance'):

    results = {}

    for file in os.listdir(f'{filepath}'):
           
        if file.endswith('.jsonl'):

            individual_result = {}
            data = pd.read_json(f'{filepath}/{file}',lines=True)
            data = data[['custom_id','response']]
            
            for i in range(len(data)):
                result = data['response'][i]['body']['choices'][0]['message']['content']
                #print(result)
                result = json.loads(result)
                #print(data["custom_id"][i])
                individual_result[f'{data["custom_id"][i]}'] = result[keyword]
                #print(individual_result)

            results[file] = individual_result
    
    #print(results)

    return results

df = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/comm_cases_valid.pkl')

df_example = df.sample(3,random_state=47)
df_main = df.drop(df_example.index)

results = combine_results('/users/sgdbareh/volatile/ECHR_Importance/Results/keyword_prediction_2', keyword='Keywords')

for name,result in results.items():
    #df_results = pd.merge(df_main,df_results,on='Filename')

    results_temp = pd.DataFrame(result.items(), columns=['Filename', 'Keywords'])
    results_temp = pd.merge(df_main,results_temp,on='Filename')
   
    y_true = results_temp['keywords_art_3']
    y_pred = results_temp['Keywords']

    #y_pred = [['350'] for i in range(len(y_pred))]

    # Initialize lists to store precision and recall for each row
    precisions = []
    recalls = []  

    for true_keywords, pred_keywords in zip(y_true, y_pred):
        true_keywords_set = set(true_keywords)
        pred_keywords_set = set(pred_keywords)

            # Convert to binary format for precision and recall calculation
        all_keywords = list(true_keywords_set.union(pred_keywords_set))
        y_true_binary = [1 if keyword in true_keywords_set else 0 for keyword in all_keywords]
        y_pred_binary = [1 if keyword in pred_keywords_set else 0 for keyword in all_keywords]
        
        #print(y_true_binary)
        # Calculate precision and recall
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)

    # Calculate average precision and recall
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)

    print(name)
    print(f"Average Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")
    print(f"F1 Score: {2 * average_precision * average_recall / (average_precision + average_recall)}")
    
