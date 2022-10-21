# Copyright (c) OpenMMLab. All rights reserved.
import os
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
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
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

    assert args.show or (args.out_img_root != '')
    # assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    pose_config = Config.fromfile(args.pose_config)
    # build the dataloader
    dataset = pose_model.cfg.data['test']['type']

    # dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    image_list = os.listdir(args.img_root)
    outputs = []
    for image_name in image_list:
        if not(image_name.endswith(".jpeg") or image_name.endswith(".jpg") or image_name.endswith(".png")):
            continue

        os.makedirs(args.out_img_root, exist_ok=True)
        out_file = os.path.join(args.out_img_root, f'vis_{image_name}')

        os.makedirs(os.path.join(args.img_root, '..', 'heatmaps'), exist_ok=True)
        heatmap_out_file = os.path.join(args.img_root, '..', 'heatmaps', f'{image_name}.npy')

        if os.path.exists(heatmap_out_file):
            continue
        print(image_name)
        image_path = os.path.join(args.img_root, image_name)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image_path)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = True

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_path,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        pose_output = {'preds': [], 'bbxes': [], 'crop_paths': [], 'bbox_ids': []}
        for r, result in enumerate(pose_results):
            pose_output['preds'].append(result['keypoints'])
            # x1, y1, x2, y2, score = result['bbox']
            pose_output['bbxes'].append(result['bbox'])
            pose_output['crop_paths'].append(args.img_root + image_name)
            pose_output['bbox_ids'].append(r)
        pose_output['preds'] = np.array(pose_output['preds'], dtype=np.float32)
        pose_output['bbxes'] = np.array(pose_output['bbxes'], dtype=np.float32)
        outputs.append(pose_output)

        # show the results
        vis_pose_result(
            pose_model,
            image_path,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

        if len(returned_outputs) == 1:
            np.save(heatmap_out_file, returned_outputs[0]['heatmap'])
        elif len(returned_outputs) == 0:
            print(f"returned_outputs has NO element for {image_name}")
        else:
            print(f"returned_outputs has more than 1 element for {image_name}")

    if args.out_img_root:
        out_path = os.path.join(args.out_img_root, "pose_detections.json")
        print(f'\nwriting results to {out_path}')
        dump(outputs, out_path)


if __name__ == '__main__':
    main()
