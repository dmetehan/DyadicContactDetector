import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

from dataset.YOUth10mClassification import SET_SPLITS

all_annot = None
stats = {}
root_dir = "/home/sac/Encfs/YOUth/10m/pci_frames/annotations/contact/cam1"
colors = {'train': 'green', 'val': 'blue', 'test': 'red'}
for annot_file in os.listdir(root_dir):
	subject = annot_file.split('.')[0]
	if subject in ["B79691", "B62414", "B65854", "B39886"]:
		# not enough annotations (<10), almost all ambiguous
		continue
	annot = pd.read_csv(os.path.join(root_dir, annot_file), header=None)
	annot.columns = ['frame', 'contact_type']
	annot['subject'] = subject
	annot = annot.reindex(columns= ['subject', 'frame', 'contact_type'])
	if all_annot is None:
		all_annot = annot
	else:
		all_annot = pd.concat((all_annot, annot)).reset_index(drop=True)
	cur_set = ''
	for _set in SET_SPLITS:
		if subject in SET_SPLITS[_set]:
			cur_set = colors[_set]
			break
	stats[subject] = {'no': len(annot[annot['contact_type'] == 0]),
					  'amb': len(annot[annot['contact_type'] == 1]),
					  'touch': len(annot[annot['contact_type'] == 2]),
					  'set': cur_set}

all_annot = all_annot[all_annot['contact_type'] != 1].reset_index(drop=True)  # removing ambiguous class
# all_annot.to_csv("YOUth_contact_annotations.csv")

# Create a DataFrame from the dictionary
df = pd.DataFrame(stats).T

# Set the subject names as the index
df.index.name = 'subject_name'
df.reset_index(inplace=True)

# Create a 'total' column
df['total'] = df['no'] + df['touch'] + df['amb']

# Create a 'ratio' column
df['ratio'] = df['touch'] / df['total']


# Sort the DataFrame by the 'ratio' column
df = df.sort_values(by='ratio',ascending=False)

print(df.head())
print(f"Contact Ratio\nMin: {df['ratio'].min()}, Max: {df['ratio'].max()}, Mean: {df['ratio'].mean()}, Std: {df['ratio'].std()}")
print(f"Video Duration\nMin: {5*df['total'].min()}, Max: {5*df['total'].max()}, Mean: {5*df['total'].mean()}, Std: {5*df['total'].std()}")
ax = df.plot.bar(x='subject_name', y='ratio', rot=0, color=df['set'], legend=False, width=1, edgecolor='black')
# Remove the x-axis ticks and their labels
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_xlabel('Parent-Infant Pairs')
ax.set_ylabel('Contact Ratio')

# Create custom patches for legend
train_patch = mpatches.Patch(color=colors['train'], label='train')
val_patch = mpatches.Patch(color=colors['val'], label='val')
test_patch = mpatches.Patch(color=colors['test'], label='test')

# Add the legend to the plot
ax.legend(handles=[train_patch, val_patch, test_patch])
plt.show()
