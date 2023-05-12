import os
import sys

import cv2

sys.path.append("/mnt/hdd1/GithubRepos/ContactClassification")
os.chdir('/mnt/hdd1/GithubRepos/ContactClassification')
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.stats import multivariate_normal
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.sampler import WeightedRandomSampler

from utils import Aug, Options, parse_config

# Images should be cropped around interacting people pairs before using this class.

class PublicYOUth10mClassification(Dataset):

    def __init__(self, root_dir, transform=None, target_transform=None, option=Options.jointmaps, target_size=(224, 224), recalc_joint_hmaps=False, bodyparts_dir=None, _set='test'):
        self._set = _set
        self.option = option
        self.resize = target_size
        self.target_size = target_size
        self.heatmaps_dir = os.path.join(root_dir, "heatmaps")
        self.gauss_hmaps_dir = os.path.join(root_dir, "gauss_hmaps")
        self.joint_hmaps_dir = os.path.join(root_dir, "joint_hmaps")
        self.crops_dir = os.path.join(root_dir, "crops")
        if bodyparts_dir:
            self.bodyparts_dir = os.path.join(root_dir, bodyparts_dir)
        os.makedirs(self.gauss_hmaps_dir, exist_ok=True)
        os.makedirs(self.joint_hmaps_dir, exist_ok=True)
        # img_labels = pd.read_csv("dataset/YOUth_contact_annotations.csv", index_col=0)
        # img_labels = img_labels[img_labels['subject'].str.contains('|'.join(set_subjects))]
        dets_file = os.path.join(root_dir, "pose_detections.json")
        img_dets = pd.read_json(dets_file)
        # img_dets = img_labels_dets[img_labels_dets['contact_type'] != 1].reset_index(drop=True)  # remove ambiguous class
        # filter only _set subjects:
        # TODO: Combine img_labels with img_dets after annotating
        self.img_labels_dets = img_dets
        # self.check_labels_dets_matching(img_labels, self.img_labels_dets)
        self.transform = transform
        self.target_transform = target_transform
        self.debug_printed = False
        self.recalc_joint_hmaps = recalc_joint_hmaps

    @staticmethod
    def check_labels_dets_matching(img_labels, img_labels_dets):
        assert len(img_labels[img_labels.duplicated(subset=['subject', 'frame', 'contact_type'])]) == 0, \
            "DUPLICATES FOUND IN img_labels"
        missing_frames = defaultdict(list)
        for index, row in img_labels.iterrows():
            crop_path = f"/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all/crops/cam1/{row['subject']}/{row['frame']}"
            if crop_path not in img_labels_dets['crop_path'].unique():
                print(crop_path)
                missing_frames[row['subject']].append(crop_path)
        assert len(missing_frames) == 0, "There are missing frames in the pose_detections.csv!"
        assert len(img_labels.drop_duplicates(subset=['subject', 'frame'])) == len(img_labels_dets)

    def __len__(self):
        return len(self.img_labels_dets)

    @staticmethod
    def bbox_xyxy2cs(x1, y1, x2, y2, aspect_ratio, padding=1.):
        """ Converts xyxy bounding box format to center and scale with the added padding.
        """
        bbox_xyxy = np.array([x1, y1, x2, y2])
        bbox_xywh = bbox_xyxy.copy()
        bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0]
        bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1]
        x, y, w, h = bbox_xywh[:4]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h], dtype=np.float32) / pixel_std
        scale = scale * padding
        return center, scale

    @staticmethod
    def bbox_cs2xyxy(center, scale, padding=1., pixel_std=200.):
        wh = scale / padding * pixel_std
        xy = center - 0.5 * wh
        bbox_xywh = np.r_[xy, wh]
        bbox_xyxy = bbox_xywh.copy()
        bbox_xyxy[2] = bbox_xyxy[2] + bbox_xyxy[0]
        bbox_xyxy[3] = bbox_xyxy[3] + bbox_xyxy[1]
        x1, y1, x2, y2 = bbox_xyxy
        return x1, y1, x2, y2

    @staticmethod
    def scale_bbox(x1, y1, x2, y2, heatmap_size, padding=1.25):
        aspect_ratio = 0.75  # fixed input size to the network: w=288, h=384
        center, scale = PublicYOUth10mClassification.bbox_xyxy2cs(x1, y1, x2, y2, aspect_ratio, padding=padding)

        hmap_aspect_ratio = heatmap_size[0] / heatmap_size[1]
        hmap_center, hmap_scale = PublicYOUth10mClassification.bbox_xyxy2cs(0, 0, heatmap_size[0], heatmap_size[1],
                                                                        hmap_aspect_ratio)

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0
        scale_x = scale[0] / heatmap_size[0]
        scale_y = scale[1] / heatmap_size[1]

        hmap_scale[0] *= scale_x
        hmap_scale[1] *= scale_y
        hmap_center[0] = center[0]
        hmap_center[1] = center[1]
        x1, y1, x2, y2 = PublicYOUth10mClassification.bbox_cs2xyxy(hmap_center, hmap_scale)
        return x1, y1, x2, y2

    def get_joint_hmaps(self, idx, rgb=False, augment=()):
        crop_path = self.img_labels_dets.loc[idx, "crop_path"]
        frame_path = crop_path.split('/')[-1]
        # label = min(self.img_labels_dets.loc[idx, "contact_type"], 1)
        label = 0
        joint_hmap_path = f'{os.path.join(self.joint_hmaps_dir, frame_path)}.npy'
        if os.path.exists(joint_hmap_path):
            joint_hmaps = np.array(np.load(joint_hmap_path), dtype=np.float32)
            if joint_hmaps.shape[1:] != self.resize:
                heatmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
                if len(self.img_labels_dets.loc[idx, 'bbxes']) > 0:
                    for p in range(len(self.img_labels_dets.loc[idx, 'bbxes'])):
                        for k in range(17):
                            heatmaps[p * 17 + k, :, :] = transforms.Resize((self.resize[0], self.resize[1]))(
                                Image.fromarray(joint_hmaps[p * 17 + k, :, :]))
                    joint_hmaps = (heatmaps - np.min(heatmaps)) / (np.max(heatmaps) - np.min(heatmaps))
                else:
                    joint_hmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
            else:
                joint_hmaps = (joint_hmaps - np.min(joint_hmaps)) / (np.max(joint_hmaps) - np.min(joint_hmaps))

            if rgb:
                crop = Image.open(crop_path)
                # noinspection PyTypeChecker
                joint_hmaps_rgb = np.concatenate((joint_hmaps,
                                                  np.transpose(np.array(crop.resize(self.resize), dtype=np.float32) / 255,
                                                               (2, 0, 1))), axis=0)
                return joint_hmaps_rgb, label
            else:
                return joint_hmaps, label
        if not self.recalc_joint_hmaps:
            return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label

        crop = Image.open(crop_path)
        heatmap_path = f'{os.path.join(self.heatmaps_dir, frame_path)}.npy'
        if not os.path.exists(heatmap_path):
            # no detections
            return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label
        hmap = np.load(heatmap_path)
        if hmap.shape[0] == 1:
            # only 1 detection
            hmap = np.concatenate(
                (hmap, np.zeros((1, hmap.shape[1] + (0 if not rgb else 3),
                                 hmap.shape[2], hmap.shape[3]), dtype=np.float32)))
        try:
            heatmap_size = hmap.shape[3], hmap.shape[2]
        except:
            print(hmap.shape)
        width, height = crop.size
        heatmaps = np.zeros((34 if not rgb else 37, height, width), dtype=np.float32)
        for p in range(len(self.img_labels_dets.loc[idx, "bbxes"])):
            x1, y1, x2, y2 = self.img_labels_dets.loc[idx, "bbxes"][p][:4]
            x1, y1, x2, y2 = self.scale_bbox(x1, y1, x2, y2, heatmap_size)
            x1, y1, x2, y2 = list(map(int, map(round, [x1, y1, x2, y2])))
            w, h = x2 - x1, y2 - y1

            rx1 = max(-x1, 0)
            ry1 = max(-y1, 0)
            rx2 = min(x2, width) - x1
            ry2 = min(y2, height) - y1
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width)
            y2 = min(y2, height)
            for k in range(17):
                heatmaps[p * 17 + k, y1:y2, x1:x2] = np.array(transforms.Resize((h, w), antialias=True)
                                                              (Image.fromarray(hmap[p, k, :, :])))[ry1:ry2, rx1:rx2]
        np.save(joint_hmap_path, heatmaps)
        label = 0
        return heatmaps, label


    def get_bodyparts(self, idx):
        part_ids = [0, 13, 18, 24, 21, 20, 11, 8, 12, 6, 2, 16, 5, 25, 22]
        bodyparts_path = f"{os.path.join(self.bodyparts_dir, self.img_labels_dets.loc[idx, 'crop_path'].split('/')[-1])}"
        if os.path.exists(bodyparts_path):
            # convert colors into boolean maps per body part channel (+background)
            bodyparts_img = np.asarray(transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(Image.open(bodyparts_path)), dtype=np.uint32)
            x = bodyparts_img // 127
            x = x * np.array([9, 3, 1])
            x = np.add.reduce(x, 2)
            bodyparts = [(x == i) for i in part_ids]
            bodyparts = np.stack(bodyparts, axis=0).astype(np.float32)
            if self.option == 0:  # debug option
                crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels_dets.loc[idx, "crop_path"]))}')
                crop = np.array(crop.resize(self.resize))
                plt.imshow(crop)
                plt.imshow(bodyparts_img, alpha=0.5)
                plt.show()
                for i in range(15):
                    plt.imshow(crop)
                    plt.imshow(bodyparts[i, :, :], alpha=0.5)
                    plt.show()
            return bodyparts
        else:
            print(f"WARNING: {bodyparts_path} doesn't exist!")
            return np.zeros((15, self.resize[0], self.resize[1]), dtype=np.float32)


    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
        if self.option == Options.debug:
            # for debugging
            if not self.debug_printed:
                print("DEBUG: ON")
                self.debug_printed = True
            data = np.zeros((52, self.resize[0], self.resize[1]), dtype=np.float32)
            label = self.img_labels_dets.loc[idx, "contact_type"]
        elif self.option == Options.jointmaps:
            data, label = self.get_joint_hmaps(idx)
        elif self.option == Options.jointmaps_rgb:
            data, label = self.get_joint_hmaps(idx, rgb=True)
        elif self.option == Options.jointmaps_rgb_bodyparts:
            data, label = self.get_joint_hmaps(idx, rgb=True)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.jointmaps_bodyparts:
            data, label = self.get_joint_hmaps(idx, rgb=False)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        else:
            raise NotImplementedError()

        return idx, data, label

def init_datasets_with_cfg(root_dir, _, cfg):
    return init_datasets(root_dir, root_dir, cfg.BATCH_SIZE, option=cfg.OPTION,
                         target_size=cfg.TARGET_SIZE, num_workers=8, bodyparts_dir=cfg.BODYPARTS_DIR)

def init_datasets_with_cfg_dict(root_dir, _, config_dict):
    return init_datasets(root_dir, root_dir, config_dict["BATCH_SIZE"], option=config_dict["OPTION"],
                         target_size=config_dict["TARGET_SIZE"], num_workers=8, bodyparts_dir=config_dict["BODYPARTS_DIR"])


def init_datasets(root_dir, _, batch_size, option=Options.jointmaps, target_size=(224, 224), num_workers=2, bodyparts_dir=None):
    test_dataset = PublicYOUth10mClassification(root_dir, option=option, target_size=target_size, bodyparts_dir=bodyparts_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader


def get_predictions():
    option = Options.jointmaps_rgb_bodyparts
    root_dir = '/mnt/hdd1/Datasets/YentlPublic'
    test_loader = init_datasets(root_dir, root_dir, batch_size=1, option=option, num_workers=1, bodyparts_dir='bodyparts_binary')
    print(len(test_loader))
    dataiter = iter(test_loader)
    count = 0
    for idx, data, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)


def get_joint_hmaps():
    option = Options.jointmaps
    root_dir = '/mnt/hdd1/Datasets/YentlPublic'
    dataset = PublicYOUth10mClassification(root_dir, option=option, recalc_joint_hmaps=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dataiter = iter(data_loader)
    count = 0
    for idx, data, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)

def get_visuals():
    option = Options.jointmaps_rgb_bodyparts
    root_dir = '/mnt/hdd1/Datasets/YentlPublic'
    dataset = PublicYOUth10mClassification(root_dir, option=option, bodyparts_dir='bodyparts_binary')
    idx = 459
    print(dataset.img_labels_dets.iloc[idx])
    joint_hmaps_rgb = dataset.get_joint_hmaps(idx, rgb=True)[0]
    bodyparts = dataset.get_bodyparts(idx)
    joint_hmaps, rgb = joint_hmaps_rgb[:34, :, :], joint_hmaps_rgb[34:, :, :]
    print(f"Joint Heatmaps: {joint_hmaps.shape}, rgb: {rgb.shape}, bodyparts: {bodyparts.shape}")
    out_dir = os.path.join(root_dir, "teaser")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(34):
        heatmap = (joint_hmaps[i, :, :] * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f"joint_hmap_{i}.jpg"), heatmap_img)

    for i in range(15):
        heatmap = (bodyparts[i, :, :] * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, f"bodyparts_hmap_{i}.jpg"), heatmap_img)

    crop = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "crop.jpg"), crop)
    # cv2.imshow('frame', crop)
    # cv2.waitKey()

    # for idx, data, label in dataiter:
    #     if "1500.jpg" in dataset.img_labels_dets.loc[idx.item(), "crop_path"]:
    #         print(idx, data.shape, dataset.img_labels_dets.loc[idx.item(), "crop_path"])



if __name__ == '__main__':
    # get_predictions()
    # get_joint_hmaps()
    get_visuals()
