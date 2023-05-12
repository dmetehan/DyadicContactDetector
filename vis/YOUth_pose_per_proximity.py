import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from utils import Options
from dataset.YOUth10mClassification import YOUth10mClassification


def draw_skeleton(image, keypoints, color):
    # List of joint pairs for the skeleton
    joint_pairs = [
        (0, 1),  # Nose - Left Eye
        (0, 2),  # Nose - Right Eye
        (1, 3),  # Left Eye - Left Ear
        (2, 4),  # Right Eye - Right Ear
        (0, 5),  # Nose - Left Shoulder
        (5, 7),  # Left Shoulder - Left Elbow
        (7, 9),  # Left Elbow - Left Wrist
        (0, 6),  # Nose - Right Shoulder
        (6, 8),  # Right Shoulder - Right Elbow
        (8, 10),  # Right Elbow - Right Wrist
        (5, 11),  # Left Shoulder - Left Hip
        (11, 13),  # Left Hip - Left Knee
        (13, 15),  # Left Knee - Left Ankle
        (6, 12),  # Right Shoulder - Right Hip
        (12, 14),  # Right Hip - Right Knee
        (14, 16)  # Right Knee - Right Ankle
    ]

    for pair in joint_pairs:
        pt1, pt2 = np.asarray(keypoints[pair[0]][:2], dtype=np.int32), np.asarray(keypoints[pair[1]][:2], dtype=np.int32)
        conf1, conf2 = keypoints[pair[0]][2], keypoints[pair[1]][2]

        if conf1 > 0.2 and conf2 > 0.2:
            cv2.line(image, tuple(pt1), tuple(pt2), color, 3)
            cv2.circle(image, tuple(pt1), 4, color, -1)
            cv2.circle(image, tuple(pt2), 4, color, -1)

    return image


def find_bin(dist):
    for b, (down, up) in enumerate(bins):
        if down < dist < up:
            return b
    return -1

ROOT_DIR = '/mnt/hdd1/GithubRepos/ContactClassification'
data = {}
for file_name in os.listdir(ROOT_DIR):
    if 'incorrect_predictions' in file_name:
        data[file_name.split('_')[-1].split('.')[0]] = (pd.read_csv(os.path.join(ROOT_DIR, file_name), header=None, names=['crop_path', 'contact_type']))
# print(data)

bg_color = (200, 200, 200)

option = Options.jointmaps_rgb_bodyparts
root_dir = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all'
_set = 'train'
dataset = YOUth10mClassification(root_dir, option=option, bodyparts_dir='bodyparts_binary', _set=_set)
all_dists = []
standard_size = 400
bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 100)]
skeleton_drawings = [[] for _ in range(len(bins))]
crops = [[] for _ in range(len(bins))]
all_bodyparts = [[] for _ in range(len(bins))]
random.seed(0)
shuffled_indices = list(range(len(dataset)))
random.shuffle(shuffled_indices)
for n in shuffled_indices:
    crop_path = dataset.img_labels_dets.loc[n, "crop_path"]

    if _set == 'test':
        if n not in [26, 29, 104, 117, 122, 143, 165, 199, 204, 230, 235, 236, 237, 239, 435, 436, 476, 485, 547, 585, 606, 624,
                     816, 1078, 1087, 1088, 1138, 1184, 1285,
                     811, 15, 1170, 37, 52, 1049, 885, 210, 1189, 1264, 359, 51, 1180, 779, 85, 89]:
            color_1 = (0, 0, 255)
            color_2 = (255, 0, 0)
        else:
            color_1 = (255, 0, 0)
            color_2 = (0, 0, 255)
    elif _set == 'train':
        if n not in [2288, 1033, 1950, 3606, 128, 2445, 1387, 960, 3591, 3050, 733, 2859, 1304, 2365, 210, 2834, 432, 2051]:
            color_1 = (0, 0, 255)
            color_2 = (255, 0, 0)
        else:
            color_1 = (255, 0, 0)
            color_2 = (0, 0, 255)
    else:
        raise ValueError()
    if len(dataset.img_labels_dets.loc[n, 'preds']) == 2:
        distances = distance_matrix(np.array(dataset.img_labels_dets.loc[n, 'preds'][0])[:, :2],
                                    np.array(dataset.img_labels_dets.loc[n, 'preds'][1])[:, :2])
        bin_ind = find_bin(np.min(distances))

        if bin_ind == -1 or len(crops[bin_ind]) >= 5:
            continue
        body_parts = cv2.imread(crop_path.replace('crops', 'bodyparts_binary'))
        crop = cv2.imread(crop_path)
        # Create a light gray background
        height, width, _ = crop.shape
        background = np.full((height, width, 3), bg_color, dtype=np.uint8)

        person1_keypoints = np.array(dataset.img_labels_dets.loc[n, 'preds'][0])
        person2_keypoints = np.array(dataset.img_labels_dets.loc[n, 'preds'][1])
        # Draw skeletons
        skeleton_image = draw_skeleton(background, person1_keypoints, color_2)  # Blue color for person 1
        skeleton_image = draw_skeleton(skeleton_image, person2_keypoints, color_1)  # Red color for person 2

        if height > width:
            scale = standard_size / height
        else:
            scale = standard_size / width
        width = int(skeleton_image.shape[1] * scale)
        height = int(skeleton_image.shape[0] * scale)
        dim = (width, height)

        delta_w = standard_size - width
        delta_h = standard_size - height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        # resize image
        skeleton_image = cv2.resize(skeleton_image, dim, interpolation=cv2.INTER_AREA)
        skeleton_image = cv2.copyMakeBorder(skeleton_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=bg_color)
        crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=bg_color)
        body_parts = cv2.resize(body_parts, dim, interpolation=cv2.INTER_AREA)
        body_parts = cv2.copyMakeBorder(body_parts, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))

        skeleton_drawings[bin_ind].append(skeleton_image)
        crops[bin_ind].append(crop)
        all_bodyparts[bin_ind].append(body_parts)
        # cv2.imshow('Skeletons', skeleton_image)
        # print(n)
        # cv2.imshow('image', cv2.imread(crop_path))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print(crop_path, 'LESS THAN 2 DETECTIONS!')
print('\n'.join(f'{dist:.2f}, {label}, {path}' for (dist, label, path) in sorted(all_dists)))


fig, axes = plt.subplots(5, 5, figsize=(10, 10))
# random.seed(0)
# random.shuffle(skeleton_drawings)
for i, ax in enumerate(axes.flat):
    if i < 5:  # If the index belongs to the first row
        ax.set_title(f'{bins[i][0]} < d < {bins[i][1]}')  # Set the column title

    ax.imshow(skeleton_drawings[i%5][i//5])
    ax.axis('off')

plt.subplots_adjust(wspace=0.01, hspace=0.008)  # Add this line to remove empty spaces between images
plt.show()


# fig, axes = plt.subplots(5, 5, figsize=(10, 10))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(cv2.cvtColor(crops[i%5][i//5], cv2.COLOR_BGR2RGB))
#     ax.axis('off')
#
# plt.subplots_adjust(wspace=0.02, hspace=0.01)  # Add this line to remove empty spaces between images
# plt.show()


fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i < 5:  # If the index belongs to the first row
        ax.set_title(f'{bins[i][0]} < d < {bins[i][1]}')  # Set the column title
    ax.imshow(cv2.cvtColor(all_bodyparts[i%5][i//5], cv2.COLOR_BGR2RGB))
    ax.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.01)  # Add this line to remove empty spaces between images
plt.show()
