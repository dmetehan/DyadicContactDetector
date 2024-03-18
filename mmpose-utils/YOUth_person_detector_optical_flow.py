# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import csv
import warnings
from collections import defaultdict
import itertools

import cv2
from PIL import Image, ImageChops
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import sys
sys.path.extend(["/mnt/hdd1/GithubRepos/DyadicContactDetector"])
from prep_crops import crop
from mmcv import Config, dump, load
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo, build_dataset

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except ModuleNotFoundError:
    print("ModuleNotFoundError")
    has_mmdet = False
except ImportError:
    print("ImportError")
    has_mmdet = False


def run_mmpose(img_dir, out_dir, subject, img_path, det_model, pose_model, args):
    global counter
    # build the dataloader
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    camera = 'cam1'
    full_optical_flow_dir = '/home/sac/Encfs/YOUth/10m/pci_frames_for_optical_flow/optical_flow'
    crop_optical_flow_dir = os.path.join(out_dir, 'optical_flow')
    heatmaps_dir = os.path.join(out_dir, 'heatmaps')
    crops_dir = os.path.join(out_dir, 'crops')
    heatmaps_dir = os.path.join(heatmaps_dir, camera)
    crops_dir = os.path.join(crops_dir, camera)
    os.makedirs(crop_optical_flow_dir, exist_ok=True)

    assert os.path.exists(heatmaps_dir) and os.path.exists(crops_dir), "Run YOUth_person_detector.py first to calculate heatmaps and crops"

    image_name = os.path.basename(img_path.replace('.png', '.jpg'))

    full_of_path = os.path.join(full_optical_flow_dir, subject, "cam1", image_name.replace('.jpg', '.png'))
    if not os.path.exists(full_of_path):
        print(f"{subject}/{image_name} optical flow does not exist.")
        return
    crop_file = os.path.join(crops_dir, subject, image_name.replace(".png", ".jpg"))
    heatmap_out_file = os.path.join(heatmaps_dir, f"{subject}_{image_name}.npy")
    crop_optical_flow_file = os.path.join(crop_optical_flow_dir, subject, image_name.replace(".jpg", ".png"))
    if os.path.exists(crop_optical_flow_file):
        return  # optical_flow already calculated
    else:
        os.makedirs(os.path.join(crop_optical_flow_dir, subject), exist_ok=True)
    assert os.path.exists(crop_file) and os.path.exists(heatmap_out_file), f'{crop_file} or {heatmap_out_file} does not exist.'

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, img_path)

    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
    # test a single image, with a list of bboxes.

    confidences = []
    for i in range(len(person_results)):
        confidences.append(person_results[i]['bbox'][-1])

    bbxes = []
    person_ids = np.argsort(confidences)[::-1][:2]
    # Get two best bounding box results and crop around them. Save the crops
    for i in person_ids:
        bbxes.append(person_results[i]['bbox'][:-1])

    if len(bbxes) == 1:
        bbxes.append(bbxes[0])
    if len(bbxes) == 0:
        # no detections
        print(f"{img_path} - no detections")
        return
    print(img_path)
    img = Image.open(img_path)
    img_crop, offset = crop(img, bbxes, [0, 1])
    dummy_file = "dummy.jpg"
    img_crop.save(dummy_file)
    img_crop = Image.open(dummy_file)
    saved_crop = Image.open(crop_file)
    diff = ImageChops.difference(img_crop.convert('RGB'), saved_crop.convert('RGB'))
    # cv2.imshow("img_crop", cv2.imread(dummy_file))
    # cv2.imshow("saved_crop", cv2.imread(crop_file))
    # cv2.waitKey(0)
    print(img_crop.size)
    print(saved_crop.size)
    print(crop_file, np.mean(diff), bbxes, offset)
    # if crop_file == "/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/all/crops/cam1/B71725/02100.jpg":
    assert img_crop.size == saved_crop.size, "crop sizes are different"
    assert not diff.getbbox(), "new crop is not the same as the saved crop"

    full_of = Image.open(full_of_path)
    crop_of = full_of.crop(offset)
    crop_of.save(crop_optical_flow_file)


def convert_folds_to_sets(folds_path):
    with open(folds_path) as f:
        folds = json.load(f)
    fold_sets = []
    for f in range(len(folds)):
        fold_sets.append({'test': folds[f],
                          'val': folds[f],
                          'train': list(itertools.chain.from_iterable([folds[i]
                                                                       for i in range(len(folds)) if i != f]))
                          })
    return fold_sets


def main():
    """Helper code for detecting people in YOUth pci_frames"""
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--set-dir', type=str, default='', help='Train or test set dir')
    parser.add_argument('--out-dir', type=str, default='', help='Train or test output dir')
    parser.add_argument('--annotation-dir', type=str, default='', help='Train or test annotation dir')
    parser.add_argument('--camera', type=str, default='', help='camera num (cam1, cam2)')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    '''
    ~/anaconda3/envs/openmmlab/bin/python /mnt/hdd1/GithubRepos/DyadicContactDetector/mmpose-utils/YOUth_person_detector_optical_flow.py
         mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py     mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth 
             mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py     mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth     
             --set-dir "/home/sac/Encfs/YOUth/10m/pci_frames/all"     --out-dir "/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/all"     --annotation-dir "/home/sac/Encfs/YOUth/10m/pci_frames/annotations/signature"     --camera "cam1"

    '''

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    pose2contact_dir = '/home/sac/GithubRepos/Pose2Contact/data/youth/signature'
    folds_path = os.path.join(pose2contact_dir, 'all', 'folds.json')

    for _set in ['train', 'test']:
        fold_dir = os.path.join(pose2contact_dir, 'fold0', f'{_set}')
        fold_sets = convert_folds_to_sets(folds_path)
        set_subjects = fold_sets[0][_set]
        labels_dets_file = os.path.join(fold_dir, "pose_detections_identity_fixed.json")
        img_labels_dets = pd.read_json(labels_dets_file)
        # filter only _set subjects:
        img_labels_dets = img_labels_dets[img_labels_dets['crop_path'].str.contains('|'.join(set_subjects))].reset_index(drop=True)
        img_labels_dets['subject'] = img_labels_dets['crop_path'].apply(lambda x: x.split('/')[-2])
        img_labels_dets['frame'] = img_labels_dets['crop_path'].apply(lambda x: x.split('/')[-1])
        for subject in img_labels_dets['subject'].unique():
            img_dir = os.path.join(args.set_dir, subject, args.camera)
            for index, row in img_labels_dets[img_labels_dets['subject'] == subject].iterrows():
                img_path = os.path.join(img_dir, row['crop_path'].split('/')[-1])
                run_mmpose(img_dir, args.out_dir, subject, img_path, det_model, pose_model, args)


if __name__ == '__main__':
    main()
