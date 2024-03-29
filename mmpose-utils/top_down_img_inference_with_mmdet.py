# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import csv
import warnings
from argparse import ArgumentParser

import numpy as np

from mmcv import Config, dump
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


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--set-dir', type=str, default='', help='Train or test set dir')
    parser.add_argument('--out-dir', type=str, default='', help='Train or test output dir')
    parser.add_argument('--img-dir', type=str, default='', help='Train or test image dir')
    parser.add_argument('--annotation-file', type=str, default='', help='csv annotation file')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--use_mmdet',
        action='store_true',
        default=False,
        help='whether to use the YOLOx bounding box detection. Uses ground-truth bounding boxes if False.')
    parser.add_argument(
        '--signatures',
        action='store_true',
        default=False,
        help='whether to use signature dataset or classification')
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

    assert has_mmdet, 'Please install mmdet to run the code.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

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

    if args.set_dir != '':
        out_img_dir = os.path.join(args.set_dir, 'pose_detections')
        heatmaps_dir = os.path.join(args.set_dir, 'heatmaps')
        labels_file = os.path.join(args.set_dir, f'crop_contact_'
                                                 f'{"classes" if not args.signatures else "signatures"}.csv')
        bbox_file = os.path.join(args.set_dir, f'interaction_contact_'
                                               f'{"classification" if not args.signatures else "signature"}.json')
        subject = "BUG"
        camera = "BUG"
    else:
        out_img_dir = os.path.join(args.out_dir, 'pose_detections')
        heatmaps_dir = os.path.join(args.out_dir, 'heatmaps')
        labels_file = args.annotation_file
        bbox_file = os.path.join(os.path.dirname(labels_file), f'interaction_contact_'
                                               f'{"classification" if not args.signatures else "signature"}.json')
        subject = os.path.basename(labels_file).split('.')[0]
        camera = labels_file.split('/')[-2]
        out_img_dir = os.path.join(out_img_dir, camera)
        heatmaps_dir = os.path.join(heatmaps_dir, camera)
    labels_reader = csv.reader(open(labels_file))
    with open(bbox_file, 'r') as f:
        bboxes_gt = json.load(f)
    outputs = []
    for crop_path, contact_type, offset in labels_reader:
        if crop_path == 'crop_path':
            continue  # skip header
        image_name = os.path.basename(crop_path)
        if not (image_name.endswith(".jpeg") or image_name.endswith(".jpg") or image_name.endswith(".png")):
            continue
        # if image_name not in ["girls_113749_00.png", "girls_113749_01.png", "girls_127333_00.png"]:
        #     continue

        if args.set_dir == '':
            os.makedirs(out_img_dir, exist_ok=True)
        if args.set_dir != '':
            out_file = os.path.join(out_img_dir, f'{image_name.replace("png", "jpg")}')
        else:
            out_file = os.path.join(out_img_dir, f'{subject}_{image_name.replace("png", "jpg")}')

        os.makedirs(heatmaps_dir, exist_ok=True)
        if args.set_dir != '':
            heatmap_out_file = os.path.join(heatmaps_dir, f'{image_name}.npy')
            print(image_name)
        else:
            heatmap_out_file = os.path.join(heatmaps_dir, f'{subject}_{image_name}.npy')
            print(f'{subject}_{image_name}')

        if args.set_dir == '':
            crop_path = os.path.join(args.img_dir, crop_path)

        if args.use_mmdet:
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, crop_path)

            # the person_results variable! (using gt instead of detection)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        else:
            # Read bounding boxes from the "interaction_contact_classification.json", subtract offset to replace
            pair_index = int(os.path.basename(crop_path).split('_')[-1].split('.')[0])
            video_name = '_'.join(os.path.basename(crop_path).split('_')[:-1])
            p_ids = bboxes_gt[video_name]['ci_classif' if not args.signatures else 'ci_sign'][pair_index]['person_ids']
            offset_x, offset_y = list(map(float, offset[1:-1].split(', ')))
            offset_arr = np.array([offset_x, offset_y, offset_x, offset_y])
            person_results = [{'bbox': np.append(np.array(bboxes_gt[video_name]['bbxes'][i] - offset_arr,
                                                          dtype=np.float32), 1.0)} for i in p_ids]
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            crop_path,
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
        assert len(pose_mean_scores) == 2, f"{len(pose_mean_scores)} detections, instead of 2"
        # TODO: THIS ARGSORT MESSES UP WITH THE ORDER OF THE REG_IDS
        ind_to_keep = [0, 1]  # np.argsort(pose_mean_scores)[::-1][:2]
        pose_output = {'preds': [], 'bbxes': [], 'crop_path': crop_path,
                       'contact_type' if args.signatures else 'reg_ids': contact_type}
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
            crop_path,
            [pose_results[i] for i in ind_to_keep],
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

        if len(returned_outputs) == 1:
            heatmaps = returned_outputs[0]['heatmap'][ind_to_keep]
            np.save(heatmap_out_file, heatmaps)
        elif len(returned_outputs) == 0:
            print(f"returned_outputs has NO element for {image_name}")
        else:
            print(f"returned_outputs has more than 1 element for {image_name}")

    if args.set_dir != "":
        out_path = os.path.join(args.set_dir, "pose_detections.json")
    else:
        out_path = os.path.join(args.out_dir, "pose_detections.json")
    print(f'\nwriting results to {out_path}')
    dump(outputs, out_path)


if __name__ == '__main__':
    main()
