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
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr


def data_process(data):
    data = data[['Filename','importance']]
    data = data.rename(columns={'importance':'real_importance'})
    return data

def combine_results(filepath,keyword = 'Case Importance',try_on=True):

    results = {}

    for file in os.listdir(f'{filepath}'):
        
        print(file)
           
        if file.endswith('.jsonl'):

            individual_result = {}
            data = pd.read_json(f'{filepath}/{file}',lines=True)
            data = data[['custom_id','response']]
            
            for i in range(len(data)):
                result = data['response'][i]['body']['choices'][0]['message']['content']
                print(result)
                if try_on:
                    try:
                        result = json.loads(result)
                    except:
                        print('can\'t convert to json')
                        result = {}
                        result[keyword] = pd.NA
                else:
                    result = json.loads(result)
                    
                #print(result)
                #print(data["custom_id"][i])
                individual_result[f'{data["custom_id"][i]}'] = result[keyword]
                #print(individual_result)

            results[file] = individual_result
    
    #print(results)

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
    
    #print(y_true)

    if experiment == 1:
        y_pred = y_pred.map({'key_case':1,'1':2,'2':3,'3':4,'I do not have enough information':0})
        
    else:
        y_pred = y_pred.map({'Committee':1,'Chamber':2,'Grand Chamber':3})
    

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

def score_results_RMSE(y_true,y_pred):
    mse = root_mean_squared_error(y_true, y_pred)
    print(f'RMSE: {mse}')
    return mse

def score_results_MAE(y_true,y_pred):
    mse = mean_absolute_error(y_true, y_pred)
    print(f'MAE: {mse}')
    return mse

def score_results_spearman(y_true,y_pred):
    spearman = spearmanr(y_true, y_pred)
    print(f'Spearman: {spearman}')
    return spearman

def confusion_matrix(y_true,y_pred, exp=1):
    
    if exp == 1:
        labels = ['Key Case', '1', '2', '3']
    else:
        labels = ['Grand Chamber', 'Chamber', 'Committee']
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues', normalize='true')
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


        