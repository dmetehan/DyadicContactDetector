import os

import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, transforms


# Images should be cropped around interacting people pairs before using this class.
class FlickrCI3DClassification(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.loc[idx, "crop_path"]}')
        img = read_image(img_path)
        label = self.img_labels.loc[idx, "contact_type"]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


def test_class():
    train_dir = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train'
    crop_dir = os.path.join(train_dir, "crops")
    labels_file = os.path.join(train_dir, "crop_contact_classes.csv")
    train_dataset = FlickrCI3DClassification(labels_file, crop_dir)
    for img, label in train_dataset:
        print(label)
        cv2.imshow("image", np.array(transforms.ToPILImage()(img))[:, :, ::-1])
        cv2.waitKey()


if __name__ == '__main__':
    test_class()
