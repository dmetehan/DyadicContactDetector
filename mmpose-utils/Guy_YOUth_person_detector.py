# Copyright (c) OpenMMLab. All rights reserved.
import os
import csv
import warnings
from PIL import Image
from argparse import ArgumentParser

import numpy as np
import sys
sys.path.extend(["/mnt/hdd1/GithubRepos/ContactClassification"])
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


def run_mmpose(img_dir, out_dir, det_model, pose_model, args, outputs, subject):
    camera = args.camera
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

    out_img_dir = os.path.join(out_dir, 'pose_det_imgs')
    out_img_dir = os.path.join(out_img_dir, camera)
    crops_dir = os.path.join(os.path.join(out_dir, 'crops'), camera)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    print(f'{subject}')
    for image_name in sorted(os.listdir(img_dir)):
        os.makedirs(os.path.join(out_img_dir, f'{subject}'), exist_ok=True)
        os.makedirs(os.path.join(crops_dir, f'{subject}'), exist_ok=True)
        out_file = os.path.join(out_img_dir, f'{subject}/{image_name.replace("png", "jpg")}')
        crop_file = os.path.join(crops_dir, f'{subject}/{image_name.replace("png", "jpg")}')
        if os.path.exists(out_file):
            raise FileNotFoundError(f'{out_file} exists')

        img_path = os.path.join(img_dir, image_name)
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
            continue

        img = Image.open(img_path)
        img_crop = crop(img, bbxes, [0, 1])
        img_crop.save(crop_file)

        # Rerun the bounding box detector on the cropped images.
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, crop_file)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        # test a single image, with a list of bboxes.


        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            crop_file,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=True,
            outputs=output_layer_names)

        pose_mean_scores = []  # this is created to keep the best 2 pose detections.
        for r, result in enumerate(pose_results):
            pose_mean_scores.append(result['keypoints'][:, 2].mean())
        ind_to_keep = np.argsort(pose_mean_scores)[::-1][:2]
        pose_output = {'preds': [], 'bbxes': [], 'image_name': image_name}
        for r in ind_to_keep:
            result = pose_results[r]
            pose_output['preds'].append(result['keypoints'])
            # x1, y1, x2, y2, score = result['bbox']
            pose_output['bbxes'].append(result['bbox'])
        pose_output['preds'] = np.array(pose_output['preds'], dtype=np.float32)
        pose_output['bbxes'] = np.array(pose_output['bbxes'], dtype=np.float32)
        outputs.append(pose_output)

        # show the results
        vis_pose_result(
            pose_model,
            crop_file,
            [pose_results[i] for i in ind_to_keep],
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)


def main():
    """Helper code for detecting people in YOUth pci_frames"""
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--set-dir', type=str, default='', help='Train or test set dir')
    parser.add_argument('--out-dir', type=str, default='', help='Train or test output dir')
    parser.add_argument('--root-dir', type=str, default='', help='Root directory')
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

    # Put these as arguments
    # args.set_dir = "/home/sac/Encfs/YOUth/10m/pci_frames/test"
    # args.out_dir = "/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/test"
    # args.annotation_dir = "/home/sac/Encfs/YOUth/10m/pci_frames/annotations/contact/test"
    # args.camera = "cam1"

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    for subject in sorted(os.listdir(args.set_dir)):
        out_path = os.path.join(args.out_dir, "pose_detections", args.camera, f"{subject}.json")
        os.makedirs(os.path.basename(out_path), exist_ok=True)
        if os.path.exists(out_path):
            print(f'{out_path} exists, extracting the remaining frames.')
            outputs = load(out_path)
        else:
            outputs = []
        img_dir = os.path.join(args.set_dir, subject, args.camera)
        """ 
        ~/anaconda3/envs/openmmlab/bin/python /mnt/hdd1/GithubRepos/ContactClassification/mmpose-utils/Guy_YOUth_person_detector.py \
        mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
        mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth \
        --set-dir "/home/sac/Encfs/YOUth/10m/pci_frames/test" --out-dir "/mnt/sdc1/Datasets/YOUth_Guy/test" --camera "cam1"
        """
        """ 
        ~/anaconda3/envs/openmmlab/bin/python /mnt/hdd1/GithubRepos/ContactClassification/mmpose-utils/Guy_YOUth_person_detector.py \
        mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
        mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth \
        --set-dir "/mnt/sdc1/Datasets/YOUth_Guy/5fps/frames" --out-dir "/mnt/sdc1/Datasets/YOUth_Guy/5fps" --camera "cam1"
        """
        run_mmpose(img_dir, args.out_dir, det_model, pose_model, args, outputs, subject)
        print(f'writing results to {out_path}')
        dump(outputs, out_path)

if __name__ == '__main__':
    main()
