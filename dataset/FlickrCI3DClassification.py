import json
import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
from utils import Aug, Options

# Images should be cropped around interacting people pairs before using this class.
class FlickrCI3DClassification(Dataset):
    def __init__(self, set_dir, transform=None, target_transform=None, option=Options.jointmaps, target_size=(224, 224), augment=(), is_test=False, bodyparts_dir=None):

        self.option = option
        if Aug.crop in augment:
            self.resize = (int(round((1.14285714286 * target_size[0]))),
                                  int(round((1.14285714286 * target_size[1]))))  # 224 to 256, 112 to 128 etc.
        else:
            self.resize = target_size
        self.target_size = target_size
        labels_file = os.path.join(set_dir, "crop_contact_classes.csv")
        dets_file = os.path.join(set_dir, "pose_detections.json")
        self.heatmaps_dir = os.path.join(set_dir, "heatmaps")
        self.gauss_hmaps_dir = os.path.join(set_dir, "gauss_hmaps")
        self.joint_hmaps_dir = os.path.join(set_dir, "joint_hmaps")
        self.crops_dir = os.path.join(set_dir, "crops")
        if bodyparts_dir:
            self.bodyparts_dir = os.path.join(set_dir, bodyparts_dir)
        os.makedirs(self.gauss_hmaps_dir, exist_ok=True)
        img_labels = pd.read_csv(labels_file)
        non_ambig_inds = img_labels.index[img_labels['contact_type'] != 1].tolist()
        self.img_labels = img_labels[img_labels['contact_type'] != 1].reset_index(drop=True)  # remove ambiguous class
        self.pose_dets = json.load(open(dets_file))
        self.pose_dets: List[Dict]
        self.pose_dets = [self.pose_dets[i] for i in non_ambig_inds]
        self.check_crop_paths(self.img_labels, self.pose_dets)
        self.transform = transform
        self.target_transform = target_transform
        self.train_inds = None  # this should be set before reading data from the dataset
        self.is_test = is_test
        self.augment = augment
        self.rand_rotate = transforms.RandomRotation(10)
        self.flip_pairs_pose = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.flip_pairs_bodyparts = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]

    def set_train_inds(self, train_inds):
        self.train_inds = train_inds

    @staticmethod
    def check_crop_paths(img_labels, pose_dets):
        for i in range(len(img_labels)):
            assert img_labels.loc[i, 'crop_path'] == pose_dets[i]['crop_path'],\
                f'{img_labels.loc[i, "crop_path"]} != {pose_dets[i]["crop_path"]}'

    def __len__(self):
        return len(self.img_labels)

    def get_gaussians(self, idx, rgb=False):
        label = self.img_labels.loc[idx, "contact_type"]
        label = min(label, 1)
        gauss_hmap_path = f'{os.path.join(self.gauss_hmaps_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}.npy'
        crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
        if os.path.exists(gauss_hmap_path):
            gauss_hmap = np.load(gauss_hmap_path)
            if gauss_hmap.shape[1:] != self.resize:
                # raise NotImplementedError("resizing gauss_hmap (34, 224, 224) not implemented!")
                heatmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
                for p in range(len(self.pose_dets[idx]['bbxes'])):
                    for k in range(17):
                        heatmaps[p*17+k, :, :] = transforms.Resize((self.resize[0], self.resize[1]))(Image.fromarray(gauss_hmap[p * 17 + k, :, :]))
                gauss_hmap = (heatmaps - np.min(heatmaps)) / (np.max(heatmaps) - np.min(heatmaps))
            else:
                gauss_hmap = (gauss_hmap - np.min(gauss_hmap)) / (np.max(gauss_hmap) - np.min(gauss_hmap))
            if rgb:
                # noinspection PyTypeChecker
                gauss_hmap_rgb = np.concatenate((gauss_hmap,
                                                 np.transpose(np.array(crop.resize(self.resize), dtype=np.float32) / 255, (2, 0, 1))), axis=0)
                return gauss_hmap_rgb, label
            else:
                return gauss_hmap, label
        return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label
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

    def get_heatmaps(self, idx, rgb=False):
        label = min(self.img_labels.loc[idx, "contact_type"], 1)
        joint_hmap_path = f'{os.path.join(self.joint_hmaps_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}.npy'
        if os.path.exists(joint_hmap_path):
            joint_hmaps = np.array(np.load(joint_hmap_path), dtype=np.float32)
            if joint_hmaps.shape[1:] != self.resize:
                heatmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
                for p in range(len(self.pose_dets[idx]['bbxes'])):
                    for k in range(17):
                        heatmaps[p*17+k, :, :] = transforms.Resize((self.resize[0], self.resize[1]))(Image.fromarray(joint_hmaps[p * 17 + k, :, :]))
                joint_hmaps = (heatmaps - np.min(heatmaps)) / (np.max(heatmaps) - np.min(heatmaps))
            else:
                joint_hmaps = (joint_hmaps - np.min(joint_hmaps)) / (np.max(joint_hmaps) - np.min(joint_hmaps))

            if rgb:
                crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
                # noinspection PyTypeChecker
                joint_hmaps_rgb = np.concatenate((joint_hmaps,
                                                 np.transpose(np.array(crop.resize(self.resize), dtype=np.float32) / 255, (2, 0, 1))), axis=0)
                return joint_hmaps_rgb, label
            else:
                return joint_hmaps, label
        return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label
        crop_path = self.img_labels.loc[idx, "crop_path"]
        crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
        heatmap_path = f'{os.path.join(self.heatmaps_dir, os.path.basename(crop_path))}.npy'
        if not os.path.exists(heatmap_path):
            # no detections
            return np.zeros((34 if not rgb else 37, 224, 224), dtype=np.float32), label
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

    def get_bodyparts(self, idx):
        part_ids = [0, 13, 18, 24, 21, 20, 11, 8, 12, 6, 2, 16, 5, 25, 22]
        bodyparts_path = f'{os.path.join(self.bodyparts_dir, "bpl_"+os.path.basename(self.img_labels.loc[idx, "crop_path"]))}'
        if os.path.exists(bodyparts_path):
            # convert colors into boolean maps per body part channel (+background)
            bodyparts_img = np.asarray(transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(Image.open(bodyparts_path)), dtype=np.uint32)
            x = bodyparts_img // 127
            x = x * np.array([9, 3, 1])
            x = np.add.reduce(x, 2)
            bodyparts = [(x == i) for i in part_ids]
            bodyparts = np.stack(bodyparts, axis=0).astype(np.float32)
            if self.option == 0:  # debug option
                crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels.loc[idx, "crop_path"]))}')
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
        if not self.is_test:
            augment = self.augment if idx in self.train_inds else ()
        else:
            augment = ()
        if idx >= len(self):
            raise IndexError()
        if self.option == 0:
            # for debugging
            print("DEBUG: ON")
            data = np.zeros((1, self.resize[0], self.resize[1]))
            label = 0
        elif self.option == Options.gaussian:
            data, label = self.get_gaussians(idx)
        elif self.option == Options.jointmaps:
            data, label = self.get_heatmaps(idx)
        elif self.option == Options.gaussian_rgb:
            data, label = self.get_gaussians(idx, rgb=True)
        elif self.option == Options.jointmaps_rgb:
            data, label = self.get_heatmaps(idx, rgb=True)
        elif self.option == Options.gaussian_rgb_bodyparts:
            data, label = self.get_gaussians(idx, rgb=True)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.jointmaps_rgb_bodyparts:
            data, label = self.get_heatmaps(idx, rgb=True)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        else:
            raise NotImplementedError()

        self.do_augmentations(data, augment)
        return data, label

    def do_augmentations(self, data, augment):
        for aug in augment:
            if aug == Aug.swap:
                if np.random.randint(2) == 0:  # 50% chance to swap
                    data[:17, :, :], data[17:34, :, :] = data[17:34, :, :], data[:17, :, :]
            elif aug == Aug.hflip:
                if np.random.randint(2) == 0:  # 50% chance to flip
                    # swap channels of left/right pairs of pose channels
                    for i, j in self.flip_pairs_pose:
                        data[i, :, :], data[j, :, :] = data[j, :, :], data[i, :, :]
                        data[i+17, :, :], data[j+17, :, :] = data[j+17, :, :], data[i+17, :, :]
                    # swap channels of left/right pairs of body-part channels
                    if self.option in [Options.gaussian_rgb_bodyparts, Options.jointmaps_rgb_bodyparts]:
                        for i, j in self.flip_pairs_bodyparts:
                            data[i+37, :, :], data[j+37, :, :] = data[j+37, :, :], data[i+37, :, :]
                    data[:, :, :] = data[:, :, ::-1]  # flip everything horizontally
            elif aug == Aug.crop:
                i = torch.randint(0, self.resize[0] - self.target_size[0] + 1, size=(1,)).item()
                j = torch.randint(0, self.resize[1] - self.target_size[1] + 1, size=(1,)).item()
                data = data[:, i:i+self.target_size[0], j:j+self.target_size[1]]
            # elif aug == Aug.rotate:
            # TODO: Implement random rotation


def init_datasets_with_cfg(train_dir, test_dir, cfg):
    return init_datasets(train_dir, test_dir, cfg.BATCH_SIZE, option=cfg.OPTION, val_split=0.2,
                         target_size=cfg.TARGET_SIZE, num_workers=2,
                         augment=cfg.AUGMENTATIONS, bodyparts_dir=cfg.BODYPARTS_DIR, stratified=cfg.STRATIFIED)

def init_datasets(train_dir, test_dir, batch_size, option=Options.debug, val_split=0.2, target_size=(224, 224), num_workers=2,
                  augment=(), bodyparts_dir=None, stratified=True):
    random_seed = 1
    train_dataset = FlickrCI3DClassification(train_dir, option=option, target_size=target_size, augment=augment, bodyparts_dir=bodyparts_dir)
    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    no_contact_inds = [i for i in indices if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 0]
    contact_inds = [i for i in indices if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 1]
    train_inds = no_contact_inds[int(np.floor(val_split * len(no_contact_inds))):] \
                 + contact_inds[int(np.floor(val_split * len(contact_inds))):]
    val_indices = no_contact_inds[:int(np.floor(val_split * len(no_contact_inds)))] \
                 + contact_inds[:int(np.floor(val_split * len(contact_inds)))]
    train_weights, val_weights = [0 for _ in range(len(indices))], [0 for _ in range(len(indices))]
    for i in train_inds:
        if stratified:
            train_weights[i] = 3.35 if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 1 else 1
        else:
            train_weights[i] = 1
    for i in val_indices:
        val_weights[i] = 1  # 3.35 if min(train_dataset.img_labels.loc[i, "contact_type"], 1) == 1 else 1
    train_dataset.set_train_inds(train_inds)
    # Creating data samplers: WeightedRandomSampler
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_inds), replacement=True)
    val_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(val_indices), replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    test_dataset = FlickrCI3DClassification(test_dir, option=option, target_size=target_size, is_test=True, bodyparts_dir=bodyparts_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader


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
    train_dataset = FlickrCI3DClassification(train_dir, option=option, bodyparts_dir="bodyparts_binary", is_test=True)
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
    option = 2
    train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    train_dataset = FlickrCI3DClassification(train_dir, option=option)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    dataiter = iter(train_loader)
    count = 0
    for heatmap, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)

    test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    test_dataset = FlickrCI3DClassification(test_dir, option=option)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    dataiter = iter(test_loader)
    count = 0
    for heatmap, label in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)


if __name__ == '__main__':
    # TODO: Test if the channels are matching (overlay poses and bodyparts on the rgb image)
    test_class()
    # test_get_heatmaps()
    # test_get_bodyparts()
