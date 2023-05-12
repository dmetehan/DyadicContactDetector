from collections import defaultdict
import json
import cv2
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

from utils import Options
from dataset.YOUth10mClassification import YOUth10mClassification

# def calc_proximity():

option = Options.jointmaps_rgb_bodyparts
root_dir = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all'
dataset_train = YOUth10mClassification(root_dir, option=option, bodyparts_dir='bodyparts_binary', _set='train')
dataset_val = YOUth10mClassification(root_dir, option=option, bodyparts_dir='bodyparts_binary', _set='val')
dataset_test = YOUth10mClassification(root_dir, option=option, bodyparts_dir='bodyparts_binary', _set='test')
# nth = 213
# print(dataset_test.img_labels_dets.loc[nth])
# print(len(dataset_test.img_labels_dets.loc[nth, 'preds']))
all_distances = defaultdict(list)
for name, dataset in [('train', dataset_train), ('val', dataset_val), ('test', dataset_test)]:
    for n in range(len(dataset)):
        if len(dataset.img_labels_dets.loc[n, 'preds']) == 2:
            distances = distance_matrix(np.array(dataset.img_labels_dets.loc[n, 'preds'][0])[:, :2],
                                        np.array(dataset.img_labels_dets.loc[n, 'preds'][1])[:, :2])
            print(distances.shape)
            print(np.min(distances))
            print(np.mean(distances))
            all_distances[name].append(np.min(distances))
        else:
            all_distances[name].append(-1)
        # rgb, _ = dataset_test.get_rgb(n)
        # crop = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
        # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # cv2.imshow("crop", crop)
        # cv2.waitKey()
        # break
# colors = [(255, 0, 0), (0, 255, 0)]
#     for pred in person_preds:
#         cv2.circle(crop, (int(pred[0]), int(pred[1])), 3, colors[p], 1)

with open("../all_distances.json", "w") as fp:
    json.dump(all_distances, fp)

counts, bins = np.histogram(all_distances['train'])
plt.stairs(counts, bins)
