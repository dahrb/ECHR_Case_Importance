import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
filepath = '/users/sgdbareh/volatile/ECHR_Importance/PREDICTION/batches/TEST_SEMANTIC_NO_RELEVANCE/final_examples_10_OPENAI-2048-COSINE_EXAMPLE_DICT.pkl'
data = pd.read_pickle(filepath)

# Reformat nested dictionary
results = {key: value for key, value in data.items()}
new_results = {key: list(value.values()) for key, value in results.items()}

# Flatten the list of integers from the dictionary values
all_ints = [val for sublist in new_results.values() for val in sublist]

# Define category labels corresponding to values 1 to 4
labels = ["Key Case", "Level 1", "Level 2", "Level 3"]

# Create histogram with bins centered on 1–4
bins = np.arange(0.5, 5.5, 1)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(all_ints, bins=bins, edgecolor='black', rwidth=0.8)

# Set custom labels
plt.xticks(ticks=range(1, 5), labels=labels)

# Labels and title
plt.xlabel('Case Level')
plt.ylabel('Frequency')
#plt.title('Distribution Without Relevance – GPT-2048')

# Save then show the plot
plt.savefig('dist_wo_rel_gpt_2048.png', format='png', dpi=300)
plt.show()

# print('plotting')
# #Plot the distribution
# hist, edges = np.histogram(all_ints_gold, bins=range(min(all_ints_gold), max(all_ints_gold) + 2))#)], edgecolor='black')
# plt.bar(labels,hist,edgecolor='black')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# #plt.xticks(ticks=all_ints_gold,labels=["Key Case", "Level 1", "Level 2", "Level 3"])
# plt.savefig('court_level.png',format='png',dpi=300)