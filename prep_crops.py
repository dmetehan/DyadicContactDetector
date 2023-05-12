# This code crops the images around the bounding boxes of each interacting people couple.
import os
import sys
import cv2
import json
import pandas as pd


# check_names = ['boys_21220', 'boys_21644', 'boys_53117']


def crop(img, bbxes, person_ids):
    try:
        height, width, _ = img.shape
    except AttributeError:
        width, height = img.size
    p = [{'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}, {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}]
    p[0]['x1'], p[0]['y1'], p[0]['x2'], p[0]['y2'] = bbxes[person_ids[0]]
    p[1]['x1'], p[1]['y1'], p[1]['x2'], p[1]['y2'] = bbxes[person_ids[1]]
    # get bounding box including two interacting people's bounding boxes tightly
    x1, y1, x2, y2 = min(p[0]['x1'], p[1]['x1']), min(p[0]['y1'], p[1]['y1']), max(p[0]['x2'], p[1]['x2']), max(
        p[0]['y2'], p[1]['y2'])

    # calculate offsets to add to the tight bounding box including two interacting people
    scale_left, scale_right, scale_top, scale_bottom = 0.11, 0.11, 0.11, 0.11
    left = 1 if p[1]['x1'] < p[0]['x1'] else 0
    right = 1 if p[1]['x2'] > p[0]['x2'] else 0
    top = 1 if p[1]['y1'] < p[0]['y1'] else 0
    bottom = 1 if p[1]['y2'] > p[0]['y2'] else 0
    dx_left = int(round((p[left]['x2'] - p[left]['x1']) * scale_left))
    dx_right = int(round((p[right]['x2'] - p[right]['x1']) * scale_right))
    dy_top = int(round((p[top]['y2'] - p[top]['y1']) * scale_top))
    dy_bottom = int(round((p[bottom]['y2'] - p[bottom]['y1']) * scale_bottom))

    # actual cropping
    x1, y1, x2, y2 = max(0, int(round(x1)) - dx_left), max(0, int(round(y1)) - dy_top), min(width, int(round(
        x2)) + dx_right), min(height, int(round(y2)) + dy_bottom)
    try:
        return img[y1:y2, x1:x2]
    except TypeError:
        return img.crop((x1, y1, x2, y2))


def prep(set_dir):
    img_dir = os.path.join(set_dir, 'images')
    crop_dir = os.path.join(set_dir, 'crops')
    annotations_file = os.path.join(set_dir, 'interaction_contact_classification.json')
    ann_data = json.load(open(annotations_file))

    labels_file = os.path.join(set_dir, 'crop_contact_classes.csv')
    labels = []

    for img_name in ann_data:
        img_path = os.path.join(img_dir, f'{img_name}.png')
        bbxes = ann_data[img_name]['bbxes']
        img = cv2.imread(img_path)
        count = 0
        for ci in ann_data[img_name]['ci_classif']:
            contact_type = ci['contact_type']
            crop_img = crop(img, bbxes, ci['person_ids'])
            crop_path = os.path.join(crop_dir, f'{img_name}_{count:02d}.png')
            cv2.imwrite(crop_path, crop_img)
            labels.append([crop_path, contact_type])
            count += 1
            assert count < 100, f"count ({count}) became 3 digits!"
    pd.DataFrame(labels, columns=['crop_path', 'contact_type']).to_csv(labels_file, index=False)


if __name__ == '__main__':
    # root_dir should include train and test folders for Flickr Classification dataset.
    # root_dir = '/mnt/hdd1/Datasets/CI3D/Flickr Classification'
    root_dir = sys.argv[1]
    prep(os.path.join(root_dir, 'test'))
    prep(os.path.join(root_dir, 'train'))
