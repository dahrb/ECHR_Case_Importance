from joblib import Parallel, delayed
import torch
import random
import time
# def process(i):
#     return i * i
    
# results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
# print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
def cosine(a,b):
    return random.randint(0,10)

unique_values_list = random.sample(range(0, 100000), 10000)
nodes = 0,1,2

node_similarity_score = {}
start_time = time.time()

def compute_similarity(i):
    node_cum_sum = 0
    for j in range(len(nodes)):
        node_cum_sum += cosine(j, i)
    return i, node_cum_sum

results = Parallel(n_jobs=-1)(delayed(compute_similarity)(i) for i in unique_values_list)
node_similarity_score = dict(results)

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

#between 0.0165 and 0.0179 seconds AVG