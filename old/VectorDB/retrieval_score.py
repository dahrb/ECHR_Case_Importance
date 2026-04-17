#To Do

#1. Calculate correlation
#2. Calculate precision/recall @ x

import pandas as pd
import pytrec_eval


real_list = {'1046596596':{'134':1,'154':1,'295':1,'629':1,'758':1,'205':1,'167':1,'536':1}}
predicted_list = {'1046596596':{'134':1,'154':1,'295':1,'629':1,'758':1}}

#P.5
evaluator = pytrec_eval.RelevanceEvaluator(real_list, {'recall.5'})
results = evaluator.evaluate(predicted_list)

print(results)

#calc correlation


 
