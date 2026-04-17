import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

def find_file_no(similar_cases, outcome_cases):

    file_numbers = {}
            
    #find fileNo and importance for most recent appNo case
    for i in similar_cases:
        cases = outcome_cases[outcome_cases['appno'].str.contains(i)]
        cases.sort_values(by='date',ascending=True,inplace=True)

        importance = cases.iloc[-1]['importance']
        fileNo = cases.iloc[-1]['File']

        file_numbers[fileNo] = importance

    return file_numbers

# Load the data
test_data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')
test_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/comm_test_summaries.pkl')

assert len(test_data) == len(test_summaries)

# Load the outcome cases
outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')
outcome_summaries = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Summarize_Cases/Results/outcome_summaries.pkl')

assert len(outcome_cases) == len(outcome_summaries)

results_gold = {}

for i in range(len(test_data)):

    similar_cases = test_data.iloc[i]['extracted_apps']

    file_numbers = find_file_no(similar_cases=similar_cases, outcome_cases=outcome_cases)

    results_gold[test_data.iloc[i]['Filename']] = file_numbers

new_results_2 = {}

for key, inner_dict in results_gold.items():
    new_values = []
    for inner_key, inner_value in inner_dict.items():
        new_values.append(inner_value)
    new_results_2[key] = new_values

# Flatten the list of integers from the dictionary values
all_ints_gold = [val for sublist in new_results_2.values() for val in sublist]
labels=["Key Case", "Level 1", "Level 2", "Level 3"]

# Histogram bin edges centered on values 1–4
bins = np.arange(0.5, 5.5, 1)  # bins: [0.5, 1.5, 2.5, 3.5, 4.5]

# Plot histogram
plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(all_ints_gold, bins=bins, edgecolor='black', rwidth=0.8)

# Set tick positions and labels
plt.xticks(ticks=range(1, 5), labels=labels)

# Labels and title
plt.xlabel('Case Level')
plt.ylabel('Frequency')
#plt.title('Distribution of Case Levels')

# Save and show
plt.savefig('gold_standard.png', format='png', dpi=300)
# print('plotting')
# #Plot the distribution
# hist, edges = np.histogram(all_ints_gold, bins=range(min(all_ints_gold), max(all_ints_gold) + 2))#)], edgecolor='black')
# plt.bar(labels,hist,edgecolor='black')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# #plt.xticks(ticks=all_ints_gold,labels=["Key Case", "Level 1", "Level 2", "Level 3"])
# plt.savefig('court_level.png',format='png',dpi=300)