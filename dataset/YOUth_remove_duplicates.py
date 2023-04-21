import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.model_selection import train_test_split


root_dir = "/home/sac/Encfs/YOUth/10m/pci_frames/annotations/contact/cam1"
out_dir = "/home/sac/Encfs/YOUth/10m/pci_frames/annotations/contact/cam1new"
for annot_file in sorted(os.listdir(root_dir)):
	subject = annot_file.split('.')[0]
	# print(subject)
	annot = pd.read_csv(os.path.join(root_dir, annot_file), header=None)
	if len(annot[annot.duplicated(subset=[0, 1])]) > 0:
		print(f'{len(annot[annot.duplicated(subset=[0, 1])])} duplicates found in {subject}')
		annot = annot.drop_duplicates(subset=[0, 1]).reset_index(drop=True)
		annot.to_csv(os.path.join(out_dir, annot_file), header=None, index=None)
