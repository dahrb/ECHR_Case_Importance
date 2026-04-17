import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

test_data = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/test.pkl')
outcome_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases_for_network.pkl')
data = test_data
file_mapping = {
    'pruned_ADMISSIBILITY_meta.json': 'Chamber',
    'pruned_CHAMBER_meta.json': 'Chamber',
    'pruned_GRANDCHAMBER_meta.json': 'Grand Chamber',
    'pruned_COMMITTEE_meta.json': 'Committee',
    'pruned_DECGRANDCHAMBER_meta.json': 'Grand Chamber',
    'pruned_ADMISSIBILITYCOM_meta.json': 'Committee',
}

data['source_file_mapped'] = data['source_file'].map(file_mapping)

# Set the style for the plot
sns.set_theme(context='paper', style="whitegrid", rc={"axes.grid": False})

# Define the order of categories
order = ['Grand Chamber', 'Chamber', 'Committee']

# Create the bar chart
plt.figure(figsize=(10, 6))
sns.countplot(x='source_file_mapped', data=data, palette='viridis', order=order, edgecolor='black')

# Add titles and labels
plt.xlabel('Court Level', fontsize=18)
plt.ylabel('Case Count', fontsize=18)
plt.xticks(fontsize=18)  # Increase the font size of the x-axis labels
plt.yticks(fontsize=18)  # Increase the font size of the y-axis labels

# Customize the plot for publication
plt.tight_layout()
plt.savefig('court_level.png',format='png',dpi=300)