import json
import os
from typing import List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, transforms
from PIL import Image, ImageDraw


# Images should be cropped around interacting people pairs before using this class.
class FlickrCI3DClassification(Dataset):
    def __init__(self, set_dir, transform=None, target_transform=None, option=1, target_size=(224, 224)):
        # option can be 1, 2, 3 or 4
        # 1: gaussian heatmaps around detected keypoints
        # 2: detected heatmaps mapped onto cropped image around interacting people
        # 3: 1 + rgb image
        # 4: 2 + rgb image
        assert option in [1, 2, 3, 4], f'option parameter {option} can be either 1, 2, 3 or 4.'
        self.option = option
        self.target_size = target_size
        labels_file = os.path.join(set_dir, "crop_contact_classes.csv")
        dets_file = os.path.join(set_dir, "pose_detections.json")
        self.heatmaps_dir = os.path.join(set_dir, "heatmaps")
        self.gauss_hmaps_dir = os.path.join(set_dir, "gauss_hmaps")
        os.makedirs(self.gauss_hmaps_dir, exist_ok=True)
        img_labels = pd.read_csv(labels_file)
        non_ambig_inds = img_labels.index[img_labels['contact_type'] != 1].tolist()
        self.img_labels = img_labels[img_labels['contact_type'] != 1].reset_index()  # remove ambiguous class
        self.pose_dets = json.load(open(dets_file))
        self.pose_dets: List[Dict]
        self.pose_dets = [self.pose_dets[i] for i in non_ambig_inds]
        self.check_crop_paths(self.img_labels, self.pose_dets)
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def check_crop_paths(img_labels, pose_dets):
        for i in range(len(img_labels)):
            assert img_labels.loc[i, 'crop_path'] == pose_dets[i]['crop_path'],\
                f'{img_labels.loc[i, "crop_path"]} != {pose_dets[i]["crop_path"]}'

    def __len__(self):
        return len(self.img_labels)

    def get_gaussians(self, idx, rgb=False, target_size=(224, 224)):
        label = self.img_labels.loc[idx, "contact_type"]
        label = min(label, 1)
        gauss_hmap_path = f'{os.path.join(self.gauss_hmaps_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}.npy'
        crop = Image.open(self.img_labels.loc[idx, "crop_path"])
        if os.path.exists(gauss_hmap_path):
            gauss_hmap = np.load(gauss_hmap_path)
            if gauss_hmap.shape[1:] != target_size:
                # raise NotImplementedError("resizing gauss_hmap (34, 224, 224) not implemented!")
                heatmaps = np.zeros((34, target_size[0], target_size[1]), dtype=np.float32)
                for p in range(len(self.pose_dets[idx]['bbxes'])):
                    for k in range(17):
                        heatmaps[p*17+k, :, :] = transforms.Resize((target_size[0], target_size[1]))(Image.fromarray(gauss_hmap[p*17+k, :, :]))
                gauss_hmap = heatmaps
            if rgb:
                # noinspection PyTypeChecker
                gauss_hmap_rgb = np.concatenate((gauss_hmap,
                                                 np.transpose(np.array(crop.resize(target_size)), (2, 0, 1))), axis=0)
                return gauss_hmap_rgb, label
            else:
                return gauss_hmap, label
        return np.zeros((34 if not rgb else 37, target_size[0], target_size[1]), dtype=np.float32), label
        width, height = crop.size
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        xx, yy = np.meshgrid(x, y)
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        heatmaps = np.zeros((34, 224, 224), dtype=np.float32)
        if len(self.pose_dets[idx]['preds']) == 0:
            return heatmaps, label
        for p in range(len(self.pose_dets[idx]['preds'])):
            for i in range(len(self.pose_dets[idx]['preds'][p])):
                m = self.pose_dets[idx]['preds'][p][i][:2]
                s = 10 * np.eye(2)
                k = multivariate_normal(mean=m, cov=s)
                zz = k.pdf(xxyy)
                # reshape and plot image
                img = Image.fromarray(zz.reshape((height, width)))
                # noinspection PyTypeChecker
                img = np.array(img.resize((224, 224)))
                heatmaps[17 * p + i, :, :] = img

        np.save(gauss_hmap_path, heatmaps)
        # noinspection PyArgumentList
        # heatmap = (heatmaps.max(axis=0)*2550).astype(np.uint8)
        # heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # print(np.max(heatmap_img), np.min(heatmap_img))
        # img = cv2.resize(cv2.imread(self.img_labels.loc[idx, "crop_path"]), (224, 224))
        # print(heatmap_img.shape, img.shape)
        # super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
        # cv2.imshow('frame', super_imposed_img)
        # cv2.waitKey()
        # plt.imshow(super_imposed_img)
        # plt.show()
        return heatmaps, label

    @staticmethod
    def scale_bbox(x1, y1, x2, y2, width, height, padding=1.25):
        bbox_xyxy = np.array([x1, y1, x2, y2])
        bbox_xywh = bbox_xyxy.copy()
        bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0]
        bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1]
        x, y, w, h = bbox_xywh[:4]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        aspect_ratio = width / height
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h], dtype=np.float32) / pixel_std
        scale = scale * padding
        wh = scale * pixel_std
        xy = center - 0.5 * wh
        bbox_xywh = np.r_[xy, wh]
        bbox_xyxy = bbox_xywh.copy()
        bbox_xyxy[2] = bbox_xyxy[2] + bbox_xyxy[0]
        bbox_xyxy[3] = bbox_xyxy[3] + bbox_xyxy[1]
        x1, y1, x2, y2 = bbox_xyxy
        return x1, y1, x2, y2

    def get_heatmaps(self, idx, rgb=False):
        # TODO: add rgb option
        crop = Image.open(self.img_labels.loc[idx, "crop_path"])
        label = min(self.img_labels.loc[idx, "contact_type"], 1)
        heatmap_path = f'{os.path.join(self.heatmaps_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}.npy'
        if not os.path.exists(heatmap_path):
            # no detections
            return np.zeros((34 if not rgb else 37, 224, 224), dtype=np.float32), label
        heatmap = np.load(heatmap_path)
        if heatmap.shape[0] == 1:
            # only 1 detection
            heatmap = np.concatenate(
                (heatmap, np.zeros((1, heatmap.shape[1] + (0 if not rgb else 3),
                                    heatmap.shape[2], heatmap.shape[3]), dtype=np.float32)))
        width, height = crop.size
        heatmaps = np.zeros((34 if not rgb else 37, height, width, 3), dtype=np.float32)
        # heatmaps = np.zeros((34 if not rgb else 37, height, width), dtype=np.float32)
        for p in range(len(self.pose_dets[idx]['bbxes'])):
            x1, y1, x2, y2 = self.pose_dets[idx]['bbxes'][p][:4]

            x1, y1, x2, y2 = self.scale_bbox(x1, y1, x2, y2, width, height, padding=1.25)
            x1, y1, x2, y2 = list(map(int, map(round, [x1, y1, x2, y2])))

            w_aug, h_aug = x2 - x1, y2 - y1
            person_hmaps = np.zeros((34 if not rgb else 37, h_aug, w_aug, 3), dtype=np.float32)
            person_crop = np.zeros((h_aug, w_aug, 3), dtype=np.float32)
            rx1 = max(-x1, 0)
            ry1 = max(-y1, 0)
            rx2 = min(x2, width) - x1
            ry2 = min(y2, height) - y1
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width)
            y2 = min(y2, height)
            # print(p, x1, y1, x2, y2)
            # print(p, rx1, ry1, rx2, ry2)
            for k in range(17):
                heatmaps[p*17+k, y1:y2, x1:x2, p] = np.array(transforms.Resize((h_aug, w_aug), antialias=True)
                                                             (Image.fromarray(heatmap[p, k, :, :])))[ry1:ry2, rx1:rx2]
                person_hmaps[p*17+k, :, :, p] = np.array(transforms.Resize((h_aug, w_aug), antialias=True)
                                                                 (Image.fromarray(heatmap[p, k, :, :])))
                # heatmaps[p*17+k, y1:y2, x1:x2] = np.array(transforms.Resize((h_aug, w_aug), antialias=True)
                #                                           (Image.fromarray(heatmap[p, k, :, :])))[ry1:ry2, rx1:rx2]
            heatmap_img = Image.fromarray((person_hmaps.max(axis=0)*255).astype(np.uint8))
            person_crop[ry1:ry2, rx1:rx2, :] = np.array(crop)[y1:y2, x1:x2, :]
            super_imposed_img = Image.blend(Image.fromarray(person_crop.astype(np.uint8)).convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
            plt.imshow(super_imposed_img)
            plt.show()

        # TODO: Pad the person crop with black regions to overlay the heatmap (to see if edges create a problem!)
        # noinspection PyArgumentList
        # print("heatmaps", heatmaps.shape)
        heatmap_img = Image.fromarray((heatmaps.max(axis=0)*255).astype(np.uint8))
        # print(heatmap_img.size, crop.size)
        super_imposed_img = Image.blend(crop.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
        draw = ImageDraw.Draw(super_imposed_img)
        for p in range(len(self.pose_dets[idx]['bbxes'])):
            draw.rectangle(self.pose_dets[idx]['bbxes'][p][:4], outline="red" if p == 0 else "green")
        for p in range(len(self.pose_dets[idx]['preds'])):
            for k in range(17):
                x, y, _ = self.pose_dets[idx]['preds'][p][k]
                draw.ellipse((x-5, y-5, x+5, y+5), outline="yellow" if p == 0 else "white")
        plt.imshow(super_imposed_img)
        # plt.savefig(f'{idx:0d}.png')
        plt.show()
        if self.transform:
            heatmap = self.transform(heatmap)
        if self.target_transform:
            label = self.target_transform(label)
        return heatmap, label

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
        if self.option == 1:
            return self.get_gaussians(idx, target_size=self.target_size)
        elif self.option == 2:
            return self.get_heatmaps(idx)
        elif self.option == 3:
            return self.get_gaussians(idx, rgb=True, target_size=self.target_size)
        elif self.option == 4:
            return self.get_heatmaps(idx, rgb=True)
        else:
            raise NotImplementedError()


def init_datasets(train_dir, test_dir, batch_size, option=1, val_split=0.2, target_size=(224, 224), num_workers=2):
    random_seed = 1
    train_dataset = FlickrCI3DClassification(train_dir, option=option, target_size=target_size)
    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    # Creating data samplers: SubsetRandomSampler
    # split = int(np.floor(val_split * dataset_size))
    # train_indices, val_indices = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    # Creating data samplers: WeightedRandomSampler
    no_contact_inds = [i for i in indices if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 0]
    contact_inds = [i for i in indices if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 1]
    train_inds = no_contact_inds[int(np.floor(val_split * len(no_contact_inds))):] \
                 + contact_inds[int(np.floor(val_split * len(contact_inds))):]
    val_indices = no_contact_inds[:int(np.floor(val_split * len(no_contact_inds)))] \
                 + contact_inds[:int(np.floor(val_split * len(contact_inds)))]
    train_weights, val_weights = [0 for _ in range(len(indices))], [0 for _ in range(len(indices))]
    for i in train_inds:
        train_weights[i] = 3.35 if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 1 else 1
    for i in val_indices:
        val_weights[i] = 1  # 3.35 if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 1 else 1
    train_sampler = WeightedRandomSampler(train_weights, len(train_inds), replacement=True)
    val_sampler = WeightedRandomSampler(val_weights, len(val_indices), replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    test_dataset = FlickrCI3DClassification(test_dir, option=option, target_size=target_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader


def test_class():
    option = 2
    train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/train'
    # train_dataset = FlickrCI3DClassification(train_dir, option=option)
    test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/test'
    # test_dataset = FlickrCI3DClassification(test_dir, option=option)
    train_loader, validation_loader, test_loader = init_datasets(train_dir, test_dir, batch_size=1, option=option, num_workers=1)
    # print(len(train_loader))
    dataiter = iter(train_loader)
    # for heatmap, label in dataiter:
    #     # continue
    #     print(np.count_nonzero(label), len(label))
    #     cv2.imshow("image", np.array(transforms.ToPILImage()(heatmap[0, 0])))
    #     cv2.waitKey()
    for heatmap, label in validation_loader:
        print(np.count_nonzero(label), len(label))
    # for heatmap, label in test_loader:
    #     continue


def test_get_heatmaps():
    option = 2
    train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/train'
    train_dataset = FlickrCI3DClassification(train_dir, option=option)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    dataiter = iter(train_loader)
    for heatmap, label in dataiter:
        print(np.count_nonzero(label), len(label))


if __name__ == '__main__':
    # test_class()
    test_get_heatmaps()
