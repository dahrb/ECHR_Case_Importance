"""
Version history
v1_2 = incorporated more experiments including Chamber v Committee v GC and using manual summarisations
v1_1 = implements more flexibility for scoring various experiments such as Court and Judgment vs Decision
v1_0 = implements result processing and scoring
"""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, multilabel_confusion_matrix, ConfusionMatrixDisplay
import os
import json
import matplotlib.pyplot as plt

def data_process(data):
    data = data[['Filename','importance']]
    data = data.rename(columns={'importance':'real_importance'})
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
    
    print(results)

    return results

def process_results(results,data,experiment=2,y_true_keyword = 'real_importance',y_pred_keyword = 'Case Importance'):
    '''
    function processes experiment 1 and 2 results which use different case importance levels to be fairly compared to the ones recorded in the dataset
    it also provides some processing
    '''
    df = pd.DataFrame(results.items(), columns=['Filename', y_pred_keyword])
    df = pd.merge(data,df,on='Filename')
    y_true = df[y_true_keyword] #real_court
    y_pred = df[y_pred_keyword] #Court
    #print(y_pred)
    if experiment == 1:
        y_pred = y_pred.map({'key_case':1,'1':2,'2':3,'3':4,'I do not have enough information':0})
        
    else:
        #y_pred = y_pred.map({1:1,2:2,3:3,4:4,'I do not have enough information':0})
        pass

    try:
        y_pred = y_pred.astype('int64')
    except:
        print('can\'t convert to int64')
    

    return y_pred, y_true


def score_results(y_true,y_pred):
    scores = precision_recall_fscore_support(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f'Precision: {scores[0]}\nRecall: {scores[1]}\nF1: {scores[2]}\nMCC: {mcc}')
    return scores,mcc

def confusion_matrix(y_true,y_pred):
    #cm = multilabel_confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,cmap='Blues',normalize='true')
    #print(disp)
    plt.show()

def process_as_binary(y):
    y = y.map({1:'Hard',2:'Hard',3:'Easy',4:'Easy'})
    return y
                  
if __name__ == '__main__':

    data = pd.read_pickle('valid_data.pkl')
    data = data_process(data[:-4])   
    
    results = combine_results('./Results/experiment_2_valid_binary')
    for name,result in results.items():
        print(name)
        y_pred, y_true = process_results(result,data,experiment=2)
        break
        #y_true = process_as_binary(y_true)
        #y_pred, y_true = process_as_binary(y_pred,y_true)
        #score_results(y_true,y_pred)
        #print('\n')


        