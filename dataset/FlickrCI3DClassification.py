import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from typing import List, Dict
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import Dataset
from scipy.stats import multivariate_normal
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.sampler import WeightedRandomSampler
from scipy.spatial import distance_matrix

from utils import Aug, Options, parse_config


# For the error "OSError: unrecognized data stream contents when reading image file"
# ImageFile.LOAD_TRUNCATED_IMAGES = True


# Images should be cropped around interacting people pairs before using this class.
class FlickrCI3DClassification(Dataset):
    def __init__(self, set_dir, transform=None, target_transform=None, option=Options.jointmaps, target_size=(224, 224),
                 augment=(), is_test=False, bodyparts_dir=None, recalc_hmaps=False, signatures=False):
        self.signatures = signatures
        target_size = tuple(target_size)
        self.option = option
        if Aug.crop in augment:
            self.resize = (int(round((1.14285714286 * target_size[0]))),
                                  int(round((1.14285714286 * target_size[1]))))  # 224 to 256, 112 to 128 etc.
        else:
            self.resize = target_size
        self.target_size = target_size
        self.recalc = recalc_hmaps
        labels_file = os.path.join(set_dir, f"crop_contact_{'classes' if not signatures else 'signatures'}.csv")
        dets_file = os.path.join(set_dir, "pose_detections.json")
        self.heatmaps_dir = os.path.join(set_dir, "heatmaps")
        self.gauss_hmaps_dir = os.path.join(set_dir, "gauss_hmaps")
        self.joint_hmaps_dir = os.path.join(set_dir, "joint_hmaps")
        self.crops_dir = os.path.join(set_dir, "crops")
        if bodyparts_dir:
            self.bodyparts_dir = os.path.join(set_dir, bodyparts_dir)
        os.makedirs(self.gauss_hmaps_dir, exist_ok=True)
        img_labels = pd.read_csv(labels_file)
        self.pose_dets = json.load(open(dets_file))
        if not signatures:
            non_ambig_inds = img_labels.index[img_labels['contact_type'] != 1].tolist()
            self.img_labels = img_labels[img_labels['contact_type'] != 1].reset_index(drop=True)  # remove ambiguous class
            self.img_labels['contact_type'] = self.img_labels['contact_type'] // 2  # convert touch labels from 2 to 1.
            self.pose_dets: List[Dict]
            self.pose_dets = [self.pose_dets[i] for i in non_ambig_inds]
        else:
            self.img_labels = img_labels
        self.check_crop_paths(self.img_labels, self.pose_dets)
        self.transform = transform
        self.target_transform = target_transform
        self.train_inds = None  # this should be set before reading data from the dataset
        self.is_test = is_test
        self.augment = augment
        self.rand_rotate = transforms.Compose([transforms.RandomRotation(10)])
        self.flip_pairs_pose = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.flip_pairs_bodyparts = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
        self.debug_printed = False
        self.color_aug = transforms.Compose([transforms.RandomPhotometricDistort()])
                                            #, transforms.GaussianBlur(3, [0.01, 1.0])])

    def set_train_inds(self, train_inds):
        self.train_inds = train_inds

    @staticmethod
    def check_crop_paths(img_labels, pose_dets):
        for i in range(len(img_labels)):
            assert img_labels.loc[i, 'crop_path'] == pose_dets[i]['crop_path'], \
                f'{img_labels.loc[i, "crop_path"]} != {pose_dets[i]["crop_path"]}'

    def __len__(self):
        return len(self.img_labels)

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
        center, scale = FlickrCI3DClassification.bbox_xyxy2cs(x1, y1, x2, y2, aspect_ratio, padding=padding)

        hmap_aspect_ratio = heatmap_size[0] / heatmap_size[1]
        hmap_center, hmap_scale = FlickrCI3DClassification.bbox_xyxy2cs(0, 0, heatmap_size[0], heatmap_size[1],
                                                                        hmap_aspect_ratio)

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0
        scale_x = scale[0] / heatmap_size[0]
        scale_y = scale[1] / heatmap_size[1]

        hmap_scale[0] *= scale_x
        hmap_scale[1] *= scale_y
        hmap_center[0] = center[0]
        hmap_center[1] = center[1]
        x1, y1, x2, y2 = FlickrCI3DClassification.bbox_cs2xyxy(hmap_center, hmap_scale)
        return x1, y1, x2, y2

    def get_label(self, idx):
        if self.signatures:
            label = json.loads(self.img_labels.loc[idx, "reg_ids"])
            label = list(set([tuple(elem) for elem in label]))
            label = [[elem[0] for elem in label], [elem[1] for elem in label]]
            label[0] = list(set(label[0]))
            label[1] = list(set(label[1]))
            onehot = [0] * 42
            for l in label[0]:
                onehot[l] = 1
            for l in label[1]:
                onehot[21+l] = 1
            return onehot
        else:
            return self.img_labels.loc[idx, "contact_type"]

    def get_heatmaps(self, idx, rgb=False):
        label = self.get_label(idx)
        joint_hmap_path = f'{os.path.join(self.joint_hmaps_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}.npy'
        if os.path.exists(joint_hmap_path):
            joint_hmaps = np.array(np.load(joint_hmap_path), dtype=np.float32)
            if idx in self.train_inds and joint_hmaps.shape[1:] != self.resize:
                heatmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
                for p in range(len(self.pose_dets[idx]['bbxes'])):
                    for k in range(17):
                        heatmaps[p*17+k, :, :] = transforms.Resize((self.resize[0], self.resize[1]))(Image.fromarray(joint_hmaps[p * 17 + k, :, :]))
                joint_hmaps = (heatmaps - np.min(heatmaps)) / (np.max(heatmaps) - np.min(heatmaps))
            elif idx not in self.train_inds and joint_hmaps.shape[1:] != self.target_size:
                heatmaps = np.zeros((34, self.target_size[0], self.target_size[1]), dtype=np.float32)
                for p in range(len(self.pose_dets[idx]['bbxes'])):
                    for k in range(17):
                        heatmaps[p*17+k, :, :] = transforms.Resize((self.target_size[0], self.target_size[1]))(Image.fromarray(joint_hmaps[p * 17 + k, :, :]))
                joint_hmaps = (heatmaps - np.min(heatmaps)) / (np.max(heatmaps) - np.min(heatmaps))
            else:
                joint_hmaps = (joint_hmaps - np.min(joint_hmaps)) / (np.max(joint_hmaps) - np.min(joint_hmaps))

            if rgb:
                crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
                # noinspection PyTypeChecker
                try:
                    joint_hmaps_rgb = np.concatenate((joint_hmaps,
                                                     np.transpose(np.array(crop.resize(self.resize if idx in self.train_inds
                                                                                       else self.target_size), dtype=np.float32) / 255, (2, 0, 1))), axis=0)
                except OSError:
                    print(self.img_labels.loc[idx, "crop_path"])
                    raise ValueError()
                return joint_hmaps_rgb, label
            else:
                return joint_hmaps, label
        if not self.recalc:
            if idx in self.train_inds:
                return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label
            else:
                return np.zeros((34 if not rgb else 37, self.target_size[0], self.target_size[1]), dtype=np.float32), label
        else:
            crop_path = self.img_labels.loc[idx, "crop_path"]
            crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
            heatmap_path = f'{os.path.join(self.heatmaps_dir, os.path.basename(crop_path))}.npy'
            if not os.path.exists(heatmap_path):
                # no detections
                # print("NO DETECTIONS", heatmap_path)
                return np.zeros((34 if not rgb else 37, 224, 224), dtype=np.float32), label
            # print(crop_path)
            hmap = np.load(heatmap_path)
            if hmap.shape[0] == 1:
                # only 1 detection
                hmap = np.concatenate(
                    (hmap, np.zeros((1, hmap.shape[1] + (0 if not rgb else 3),
                                        hmap.shape[2], hmap.shape[3]), dtype=np.float32)))
            heatmap_size = hmap.shape[3], hmap.shape[2]
            width, height = crop.size
            heatmaps = np.zeros((34 if not rgb else 37, height, width), dtype=np.float32)
            for p in range(len(self.pose_dets[idx]['bbxes'])):
                x1, y1, x2, y2 = self.pose_dets[idx]['bbxes'][p][:4]
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
                # person_hmaps = np.zeros((34 if not rgb else 37, h, w, 3), dtype=np.float32)
                # person_crop = np.zeros((h, w, 3), dtype=np.float32)
                for k in range(17):
                    heatmaps[p*17+k, y1:y2, x1:x2] = np.array(transforms.Resize((h, w), antialias=True)
                                                              (Image.fromarray(hmap[p, k, :, :])))[ry1:ry2, rx1:rx2]
                    # person_hmaps[p*17+k, :, :, p] = np.array(transforms.Resize((h, w), antialias=True)
                    #                                                  (Image.fromarray(heatmap[p, k, :, :])))
                # heatmap_img = Image.fromarray((person_hmaps.max(axis=0)*255).astype(np.uint8))
                # person_crop[ry1:ry2, rx1:rx2, :] = np.array(crop)[y1:y2, x1:x2, :]
                # super_imposed_img = Image.blend(Image.fromarray(person_crop.astype(np.uint8)).convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
                # plt.imshow(super_imposed_img)
                # plt.show()
            # noinspection PyArgumentList
            # heatmap_img = Image.fromarray((heatmaps.max(axis=0)*255).astype(np.uint8))
            # super_imposed_img = Image.blend(crop.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
            # draw = ImageDraw.Draw(super_imposed_img)
            # for p in range(len(self.pose_dets[idx]['bbxes'])):
            #     draw.rectangle(self.pose_dets[idx]['bbxes'][p][:4], outline="red" if p == 0 else "green")
            # for p in range(len(self.pose_dets[idx]['preds'])):
            #     for k in range(17):
            #         x, y, _ = self.pose_dets[idx]['preds'][p][k]
            #         draw.ellipse((x-5, y-5, x+5, y+5), outline="yellow" if p == 0 else "white")
            # plt.imshow(super_imposed_img)
            # plt.savefig(f'{idx:0d}.png')
            # plt.show()
            np.save(joint_hmap_path, heatmaps)
            return heatmaps, label

    def get_rgb(self, idx):
        label = self.get_label(idx)
        crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
        try:
            crop = np.transpose(np.array(crop.resize(self.resize if idx in self.train_inds
                                                     else self.target_size), dtype=np.float32) / 255, (2, 0, 1))
        except OSError:
            print(self.img_labels.loc[idx, "crop_path"])
            raise ValueError()

        return crop, label

    def get_bodyparts(self, idx):
        part_ids = [0, 13, 18, 24, 21, 20, 11, 8, 12, 6, 2, 16, 5, 25, 22]
        bodyparts_path = f'{os.path.join(self.bodyparts_dir, "bpl_"+os.path.basename(self.img_labels.loc[idx, "crop_path"]))}'
        bodyparts_base_path = f'{bodyparts_path.split(".")[0]}'
        if os.path.exists(bodyparts_path):
            # convert colors into boolean maps per body part channel (+background)
            bodyparts_img = np.asarray(transforms.Resize(self.resize if idx in self.train_inds else self.target_size, interpolation=InterpolationMode.NEAREST)(Image.open(bodyparts_path)), dtype=np.uint32)
            x = bodyparts_img // 127
            x = x * np.array([9, 3, 1])
            x = np.add.reduce(x, 2)
            bodyparts = [(x == i) for i in part_ids]
            bodyparts = np.stack(bodyparts, axis=0).astype(np.float32)
            if self.option == 0:  # debug option
                crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
                crop = np.array(crop.resize(self.resize if idx in self.train_inds else self.target_size))
                plt.imshow(crop)
                plt.imshow(bodyparts_img, alpha=0.5)
                plt.show()
                for i in range(15):
                    plt.imshow(crop)
                    plt.imshow(bodyparts[i, :, :], alpha=0.5)
                    plt.show()
            return bodyparts
        elif os.path.exists(bodyparts_base_path + '_0.png'):
            bodyparts = np.zeros((self.resize if idx in self.train_inds else self.target_size) + (15,), dtype=np.float32)
            for i in range(5):
                cur_part_path = f"{bodyparts_base_path}_{i}.png"
                # convert colors into boolean maps per body part channel (+background)
                bodyparts_img = np.asarray(
                    transforms.Resize(self.resize if idx in self.train_inds else self.target_size, interpolation=InterpolationMode.NEAREST)(Image.open(cur_part_path)),
                    dtype=np.uint32)
                bodyparts[:, :, (3*i):(3*i+3)] = bodyparts_img / 255.0  # normalization
                # if self.option == Options.debug:  # debug option
                # crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels_dets.loc[idx, "crop_path"]))}')
                # crop = np.array(crop.resize(self.resize))
                # plt.imshow(crop)
                # plt.imshow(bodyparts_img, alpha=0.5)
                # plt.show()
            return np.transpose(bodyparts, (2, 0, 1))  # reorder dimensions
        else:
            print(f"WARNING: {bodyparts_path} doesn't exist!")
            if idx in self.train_inds:
                return np.zeros((15, self.resize[0], self.resize[1]), dtype=np.float32)
            else:
                return np.zeros((15, self.target_size[0], self.target_size[1]), dtype=np.float32)

    def __getitem__(self, idx):
        if not self.is_test:
            augment = self.augment if idx in self.train_inds else ()
        else:
            augment = ()
        if idx >= len(self):
            raise IndexError()
        if self.option == Options.debug:
            # for debugging
            if not self.debug_printed:
                print("DEBUG: ON")
                self.debug_printed = True
            if idx in self.train_inds:
                data = np.zeros((52, self.resize[0], self.resize[1]), dtype=np.float32)
            else:
                data = np.zeros((52, self.target_size[0], self.target_size[1]), dtype=np.float32)
            label = self.img_labels.loc[idx, "contact_type"]
        elif self.option == Options.jointmaps:
            data, label = self.get_heatmaps(idx)
        elif self.option == Options.jointmaps_rgb:
            data, label = self.get_heatmaps(idx, rgb=True)
        elif self.option == Options.jointmaps_rgb_bodyparts:
            data, label = self.get_heatmaps(idx, rgb=True)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.jointmaps_bodyparts:
            data, label = self.get_heatmaps(idx, rgb=False)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.rgb:
            data, label = self.get_rgb(idx)
        else:
            raise NotImplementedError()

        data = self.do_augmentations(data, augment)
        return idx, data, torch.tensor(label)

    def do_augmentations(self, data, augment):
        for aug in augment:
            if aug == Aug.swap:
                if np.random.randint(2) == 0:  # 50% chance to swap
                    data[:17, :, :], data[17:34, :, :] = data[17:34, :, :], data[:17, :, :]
            elif aug == Aug.hflip:
                if np.random.randint(2) == 0:  # 50% chance to flip
                    # swap channels of left/right pairs of pose channels
                    if self.option not in [Options.rgb]:
                        for i, j in self.flip_pairs_pose:
                            data[i, :, :], data[j, :, :] = data[j, :, :], data[i, :, :]
                            data[i+17, :, :], data[j+17, :, :] = data[j+17, :, :], data[i+17, :, :]
                    # swap channels of left/right pairs of body-part channels
                    if self.option in [Options.jointmaps_rgb_bodyparts]:
                        for i, j in self.flip_pairs_bodyparts:
                            data[i+37, :, :], data[j+37, :, :] = data[j+37, :, :], data[i+37, :, :]
                    data[:, :, :] = data[:, :, ::-1]  # flip everything horizontally
            elif aug == Aug.crop:
                i = torch.randint(0, self.resize[0] - self.target_size[0] + 1, size=(1,)).item()
                j = torch.randint(0, self.resize[1] - self.target_size[1] + 1, size=(1,)).item()
                data = data[:, i:i+self.target_size[0], j:j+self.target_size[1]]
            elif aug == Aug.color:
                # random color-based augmentations to the rgb channels
                if self.option in [Options.jointmaps_rgb, Options.jointmaps_rgb_bodyparts]:
                    data[34:37, :, :] = np.transpose(self.color_aug(Image.fromarray(np.transpose(255 * data[34:37, :, :],
                                                                                                 (1, 2, 0)).astype(np.uint8))),
                                                     (2, 0, 1)).astype(np.float32) / 255
                elif self.option in [Options.rgb]:
                    data = np.transpose(self.color_aug(Image.fromarray(np.transpose(255 * data,
                                                                                    (1, 2, 0)).astype(np.uint8))),
                                                     (2, 0, 1)).astype(np.float32) / 255
            elif aug == Aug.rotate:
                if self.option in [Options.rgb]:
                    data = np.transpose(self.rand_rotate(Image.fromarray(np.transpose(255 * data,
                                                                                      (1, 2, 0)).astype(np.uint8))),
                                                     (2, 0, 1)).astype(np.float32) / 255

        return data


def init_datasets_with_cfg(train_dir, test_dir, cfg, signatures=False):
    return init_datasets(train_dir, test_dir, cfg.BATCH_SIZE, option=cfg.OPTION, val_split=0.2,
                         target_size=cfg.TARGET_SIZE, num_workers=8,
                         augment=cfg.AUGMENTATIONS, bodyparts_dir=cfg.BODYPARTS_DIR, stratified=cfg.STRATIFIED,
                         signatures=signatures)


def init_datasets_with_cfg_dict(train_dir, test_dir, config_dict):
    return init_datasets(train_dir, test_dir, config_dict["BATCH_SIZE"], option=config_dict["OPTION"], val_split=0.2,
                         target_size=config_dict["TARGET_SIZE"], num_workers=8,
                         augment=config_dict["AUGMENTATIONS"], bodyparts_dir=config_dict["BODYPARTS_DIR"],
                         stratified=config_dict["STRATIFIED"])


def init_datasets(train_dir, test_dir, batch_size, option=Options.debug, val_split=0.2, target_size=(224, 224),
                  num_workers=2, augment=(), bodyparts_dir=None, stratified=True, signatures=False):
    random_seed = 1
    train_dataset = FlickrCI3DClassification(train_dir, option=option, target_size=target_size, augment=augment,
                                             bodyparts_dir=bodyparts_dir, signatures=signatures)
    # Creating data indices for training and validation splits:
    indices = list(train_dataset.img_labels.index)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if not signatures:
        no_contact_inds = list(train_dataset.img_labels[train_dataset.img_labels["contact_type"] == 0].index)
        contact_inds = list(train_dataset.img_labels[train_dataset.img_labels["contact_type"] == 1].index)
        np.random.shuffle(no_contact_inds)
        np.random.shuffle(contact_inds)
        train_inds = no_contact_inds[int(np.floor(val_split * len(no_contact_inds))):] \
                     + contact_inds[int(np.floor(val_split * len(contact_inds))):]
        val_indices = no_contact_inds[:int(np.floor(val_split * len(no_contact_inds)))] \
                     + contact_inds[:int(np.floor(val_split * len(contact_inds)))]
    else:
        np.random.shuffle(indices)
        train_inds = indices[int(np.floor(val_split * len(indices))):]
        val_indices = indices[:int(np.floor(val_split * len(indices)))]
    train_weights, val_weights = [0 for _ in range(len(indices))], [0 for _ in range(len(indices))]
    for i in train_inds:
        if stratified and not signatures:
            train_weights[i] = 3.35 if train_dataset.img_labels.loc[i, "contact_type"] == 1 else 1
        else:
            train_weights[i] = 1
    for i in val_indices:
        val_weights[i] = 1
    train_dataset.set_train_inds(train_inds)
    # Creating data samplers: WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_inds), replacement=True)
    val_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(val_indices), replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler,
                                                    num_workers=num_workers)

    test_dataset = FlickrCI3DClassification(test_dir, option=option, target_size=target_size, is_test=True,
                                            bodyparts_dir=bodyparts_dir, signatures=signatures)
    test_dataset.set_train_inds([])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    return train_loader, validation_loader, test_loader


def test_visually(data):
    heatmaps = data[:34, :, :].numpy()
    rgb = Image.fromarray(np.transpose(255 * data[34:37, :, :].numpy(), (1, 2, 0)).astype(np.uint8))
    bodyparts = data[38:, :, :].numpy()  # ignoring the background map for visualization purposes at index 37
    heatmap_img = Image.fromarray((heatmaps.max(axis=0)*255).astype(np.uint8))
    super_imposed_img = Image.blend(rgb.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
    plt.imshow(super_imposed_img)
    plt.show()
    bodyparts_img = Image.fromarray((bodyparts.max(axis=0)*255).astype(np.uint8))
    super_imposed_img = Image.blend(rgb.convert('RGBA'), bodyparts_img.convert('RGBA'), alpha=0.5)
    plt.imshow(super_imposed_img)
    plt.show()


def test_input(cfg, train_dir, test_dir):
    # Check if the prepared input makes sense!
    cfg.OPTION = Options.jointmaps_rgb_bodyparts
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(train_dir, test_dir, cfg)
    for loader_name, data_loader in zip(["TRAINING", "VALIDATION", "TEST"], [train_loader, validation_loader, test_loader]):
        print(f"Visualizing images from the {loader_name} set:")
        for _, data, labels in data_loader:
            for datum, label in zip(data, labels):
                print("TOUCH" if label else "NO TOUCH")
                test_visually(datum)
            break


def test_overlaps(cfg, train_dir, test_dir):
    # Check if train_loader samples are not overlapping with validation_loader
    train_loader, validation_loader, _ = init_datasets_with_cfg(train_dir, test_dir, cfg)
    validation_indices = [idx.item() for indices, _, _ in validation_loader for idx in indices]
    print(validation_indices)
    train_indices = [idx.item() for indices, _, _ in train_loader for idx in indices]
    print(train_indices)
    assert len(set(train_indices).intersection(set(validation_indices))) == 0
    print(len(train_indices), len(set(train_indices)))


def test_distribution(cfg, train_dir, test_dir):
    # Check the class distribution in validation and test sets
    _, validation_loader, test_loader = init_datasets_with_cfg(train_dir, test_dir, cfg)
    validation_labels = [label.item() for _, _, labels in validation_loader for label in labels]
    touch = np.count_nonzero(validation_labels)
    no_touch = len(validation_labels) - touch
    print(f"VALIDATION - Touch: {touch} ({100 * touch / len(validation_labels):.2f}%), "
          f"No Touch: {no_touch} ({100 * no_touch / len(validation_labels):.2f}%)")
    val_touch_percent = 100 * touch / len(validation_labels)
    test_labels = [label.item() for _, _, labels in test_loader for label in labels]
    touch = np.count_nonzero(test_labels)
    no_touch = len(test_labels) - touch
    print(f"TEST - Touch: {touch} ({100 * touch / len(test_labels):.2f}%), "
          f"No Touch: {no_touch} ({100 * no_touch / len(test_labels):.2f}%)")
    test_touch_percent = 100 * touch / len(test_labels)
    assert abs(val_touch_percent - test_touch_percent) < 1


def test_distance(cfg, train_dir, test_dir):
    val_split = 0.2
    # Check the distances between people in validation and test sets
    train_dataset = FlickrCI3DClassification(train_dir, option=cfg.OPTION, target_size=cfg.TARGET_SIZE, bodyparts_dir="bodyparts_binary")
    indices = list(train_dataset.img_labels.index)
    no_contact_inds = list(train_dataset.img_labels[train_dataset.img_labels["contact_type"] == 0].index)
    contact_inds = list(train_dataset.img_labels[train_dataset.img_labels["contact_type"] == 1].index)
    np.random.shuffle(no_contact_inds)
    np.random.shuffle(contact_inds)
    train_inds = no_contact_inds[int(np.floor(val_split * len(no_contact_inds))):] \
                 + contact_inds[int(np.floor(val_split * len(contact_inds))):]
    val_indices = no_contact_inds[:int(np.floor(val_split * len(no_contact_inds)))] \
                 + contact_inds[:int(np.floor(val_split * len(contact_inds)))]
    proximity_dist = {0: [], 1: []}
    for n in val_indices:
        label = int(train_dataset.img_labels.loc[n, 'contact_type'])
        if len(train_dataset.pose_dets[n]['preds']) == 2:
            distances = distance_matrix(np.array(train_dataset.pose_dets[n]['preds'][0])[:, :2],
                                        np.array(train_dataset.pose_dets[n]['preds'][1])[:, :2])
            proximity_dist[label].append(np.min(distances))
    # print(proximity_dist)
    print("Validation set")
    print(np.mean(proximity_dist[0]), np.std(proximity_dist[0]), np.min(proximity_dist[0]), np.max(proximity_dist[0]))
    print(np.mean(proximity_dist[1]), np.std(proximity_dist[1]), np.min(proximity_dist[1]), np.max(proximity_dist[1]))
    proximity_dist = {0: [], 1: []}
    for n in train_inds:
        label = int(train_dataset.img_labels.loc[n, 'contact_type'])
        if len(train_dataset.pose_dets[n]['preds']) == 2:
            distances = distance_matrix(np.array(train_dataset.pose_dets[n]['preds'][0])[:, :2],
                                        np.array(train_dataset.pose_dets[n]['preds'][1])[:, :2])
            proximity_dist[label].append(np.min(distances))
    # print(proximity_dist)
    print("Training set")
    print(np.mean(proximity_dist[0]), np.std(proximity_dist[0]), np.min(proximity_dist[0]), np.max(proximity_dist[0]))
    print(np.mean(proximity_dist[1]), np.std(proximity_dist[1]), np.min(proximity_dist[1]), np.max(proximity_dist[1]))

    dataset = FlickrCI3DClassification(test_dir, option=cfg.OPTION, bodyparts_dir="bodyparts_binary")
    # print(dataset.pose_dets[0]['preds'])
    proximity_dist = {0: [], 1: []}
    for n in range(len(dataset)):
        label = int(dataset.img_labels.loc[n, 'contact_type'])
        if len(dataset.pose_dets[n]['preds']) == 2:
            distances = distance_matrix(np.array(dataset.pose_dets[n]['preds'][0])[:, :2],
                                        np.array(dataset.pose_dets[n]['preds'][1])[:, :2])
            proximity_dist[label].append(np.min(distances))
    # print(proximity_dist)
    print("Test set")
    print(np.mean(proximity_dist[0]), np.std(proximity_dist[0]), np.min(proximity_dist[0]), np.max(proximity_dist[0]))
    print(np.mean(proximity_dist[1]), np.std(proximity_dist[1]), np.min(proximity_dist[1]), np.max(proximity_dist[1]))


def test_class():
    option = Options.jointmaps
    train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    # train_dataset = FlickrCI3DClassification(train_dir, option=option)
    test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    # test_dataset = FlickrCI3DClassification(test_dir, option=option)
    train_loader, validation_loader, test_loader = init_datasets(train_dir, test_dir, batch_size=1, option=option, num_workers=1, bodyparts_dir="bodyparts_binary")
    # print(len(train_loader))
    # dataiter = iter(train_loader)
    # for heatmap, label in dataiter:
    #     # continue
    #     print(np.count_nonzero(label), len(label))
    #     cv2.imshow("image", np.array(transforms.ToPILImage()(heatmap[0, 0])))
    #     cv2.waitKey()
    for heatmap, label in validation_loader:
        print(np.count_nonzero(label), len(label))
    # for heatmap, label in test_loader:
    #     continue


def test_get_bodyparts():
    option = 2

    train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    train_dataset = FlickrCI3DClassification(train_dir, option=option, bodyparts_dir="bodyparts_binary")
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=4)
    dataiter = iter(train_loader)
    count = 0
    for heatmap, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)

    test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    test_dataset = FlickrCI3DClassification(test_dir, option=option, bodyparts_dir="bodyparts_binary",
                                            is_test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    dataiter = iter(test_loader)
    count = 0
    for heatmap, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)


def test_get_heatmaps():
    option = Options.jointmaps
    # train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    # train_dir = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/train'
    # train_dataset = FlickrCI3DClassification(train_dir, option=option, recalc_hmaps=True)
    # train_dataset.set_train_inds(list(range(len(train_dataset))))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    # dataiter = iter(train_loader)
    # count = 0
    # for _, heatmap, label in dataiter:
    #     count += len(label)
    #     if count % 100 == 0:
    #         print(count)

    # test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    test_dir = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/test'
    test_dataset = FlickrCI3DClassification(test_dir, option=option, recalc_hmaps=True)
    test_dataset.set_train_inds([])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    dataiter = iter(test_loader)
    count = 0
    for _, heatmap, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)


def main():
    train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)
    # test_overlaps(cfg, train_dir_ssd, test_dir_ssd)
    # test_distribution(cfg, train_dir_ssd, test_dir_ssd)
    # test_distance(cfg, train_dir_ssd, test_dir_ssd)
    test_input(cfg, train_dir_ssd, test_dir_ssd)


if __name__ == '__main__':
    main()
    # test_class()
    # test_get_heatmaps()
    # test_get_bodyparts()
