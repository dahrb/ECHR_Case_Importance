import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test_data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')
outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases_for_network.pkl')

plt.figure(figsize=(10, 6))
sns.countplot(x='importance', data=test_data, edgecolor='black')

# Add titles and labels
#plt.title('Distribution of Importance Levels in Test Data', fontsize=16)
plt.xlabel('Importance Level', fontsize=18)
plt.ylabel('Case Count', fontsize=18)
plt.xticks(range(4), labels=['Key Case', '1', '2', '3'])
plt.xticks(fontsize=18)  # Increase the font size of the x-axis labels
plt.yticks(fontsize=18)  # Increase the font size of the y-axis labels

# Customize the plot for publication
plt.tight_layout()
plt.savefig('distribution.png',format='png',dpi=300)