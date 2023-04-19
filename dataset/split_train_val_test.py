import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.model_selection import train_test_split


stats = {}
root_dir = "/home/sac/Encfs/YOUth/10m/pci_frames/annotations/contact/cam1"
for annot_file in os.listdir(root_dir):
	subject = annot_file.split('.')[0]
	if subject in ["B79691", "B62414", "B65854", "B39886"]:
		# not enough annotations (<10), almost all ambiguous
		continue
	annot = pd.read_csv(os.path.join(root_dir, annot_file), header=None)
	stats[subject] = {'no': len(annot[annot[1] == 0]),
					  'amb': len(annot[annot[1] == 1]),
					  'touch': len(annot[annot[1] == 2])}

# Create a DataFrame from the dictionary
df = pd.DataFrame(stats).T

# Set the subject names as the index
df.index.name = 'subject_name'
df.reset_index(inplace=True)

# Create a 'ratio' column
df['ratio'] = df['no'] / (df['no'] + df['touch'])

# Sort the DataFrame by the 'ratio' column
df = df.sort_values(by='ratio')

# Assign data points to each set in a round-robin fashion
num_splits = 9
split_indices = np.arange(len(df)) % num_splits
train_indices = np.where((split_indices == 0) | (split_indices == 1) | (split_indices == 4) | (split_indices == 7) | (split_indices == 8))[0]
val_indices = np.where((split_indices == 2) | (split_indices == 6))[0]
test_indices = np.where((split_indices == 3) | (split_indices == 5))[0]
print(train_indices)

train_df = df.iloc[train_indices]
val_df = df.iloc[val_indices]
test_df = df.iloc[test_indices]

train_keys = train_df['subject_name'].tolist()
val_keys = val_df['subject_name'].tolist()
test_keys = test_df['subject_name'].tolist()

print('Training keys:', train_keys)
print('Validation keys:', val_keys)
print('Test keys:', test_keys)

# Create a scatter plot with different colors for each set
df['set'] = 'test'
df.loc[train_indices, 'set'] = 'train'
df.loc[val_indices, 'set'] = 'validation'

fig = px.scatter(df, x='no', y='touch', hover_data=['subject_name'],
                 color='set',
                 title='Scatter plot of "no" and "touch" distributions',
                 color_discrete_map={'train': 'red', 'validation': 'green', 'test': 'blue'})

fig.update_traces(marker=dict(size=12, line=dict(width=2)), selector=dict(mode='markers'))

fig.show()
pio.write_html(fig, file="scatter_plot.html")

train_total = sum([stats[key]['no'] + stats[key]['touch'] for key in train_keys])
val_total = sum([stats[key]['no'] + stats[key]['touch'] for key in val_keys])
test_total = sum([stats[key]['no'] + stats[key]['touch'] for key in test_keys])


print("Train size:\t", len(train_keys))
print("Val size:\t", len(val_keys))
print("Test size:\t", len(test_keys))
print()

print("Train no touch:\t", sum([stats[key]['no'] for key in train_keys]))
print("Val no touch:\t", sum([stats[key]['no'] for key in val_keys]))
print("Test no touch:\t", sum([stats[key]['no'] for key in test_keys]))
print()

print("Train touch:\t", sum([stats[key]['touch'] for key in train_keys]))
print("Val touch:\t", sum([stats[key]['touch'] for key in val_keys]))
print("Test touch:\t", sum([stats[key]['touch'] for key in test_keys]))
print()

print("Train no touch ratio:\t", sum([stats[key]['no'] for key in train_keys])/train_total)
print("Val no touch ratio:\t", sum([stats[key]['no'] for key in val_keys])/val_total)
print("Test no touch ratio:\t", sum([stats[key]['no'] for key in test_keys])/test_total)
print()

print("Train touch ratio:\t", sum([stats[key]['touch'] for key in train_keys])/train_total)
print("Val touch ratio:\t", sum([stats[key]['touch'] for key in val_keys])/val_total)
print("Test touch ratio:\t", sum([stats[key]['touch'] for key in test_keys])/test_total)
