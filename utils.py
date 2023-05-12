import math
import os
import cv2
import glob

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Aug.swap: Swapping the order of pose detections between people (50% chance)
#           NOT WORKING WELL because I believe the order is always top left to bottom right and switching it randomly doesn't help
# Aug.hflip: Horizontally flipping the rgb image as well as flipping left/right joints
# Aug.crop: Cropping
class Aug:
    hflip = 'hflip'
    crop = 'crop'
    rotate = 'rotate'
    swap = 'swap'
    color = 'color'
    all = {'hflip': hflip, 'crop': crop, 'rotate': rotate, 'swap': swap, 'color': color}

class Options:
    debug = "debug"
    rgb = "rgb"
    bodyparts = "bodyparts"
    gaussian = "gaussian"  # gaussian heatmaps around detected keypoints
    jointmaps = "jointmaps"  # detected heatmaps mapped onto cropped image around interacting people
    gaussian_rgb = "gaussian_rgb"
    jointmaps_rgb = "jointmaps_rgb"
    rgb_bodyparts = "rgb_bodyparts"
    jointmaps_bodyparts = "jointmaps_bodyparts"
    gaussian_rgb_bodyparts = "gaussian_rgb_bodyparts"
    jointmaps_rgb_bodyparts = "jointmaps_rgb_bodyparts"
    all = {"debug": debug, "rgb": rgb, "bodyparts": bodyparts, "gaussian": gaussian, "jointmaps": jointmaps, "gaussian_rgb": gaussian_rgb,
           "jointmaps_rgb": jointmaps_rgb, "rgb_bodyparts": rgb_bodyparts, "jointmaps_bodyparts": jointmaps_bodyparts,
           "gaussian_rgb_bodyparts": gaussian_rgb_bodyparts, "jointmaps_rgb_bodyparts": jointmaps_rgb_bodyparts}


def check_config(cfg):
    print(cfg.AUGMENTATIONS)
    assert cfg.OPTION in Options.all
    for aug in cfg.AUGMENTATIONS:
        assert aug in Aug.all

def parse_config(config_file):
    class Config:
        pass
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg_obj = Config()
    for section in cfg:
        print(section, cfg[section])
        setattr(cfg_obj, section, cfg[section])
    setattr(cfg_obj, "ID", config_file.split('/')[-1].split('_')[0])
    check_config(cfg_obj)
    return cfg_obj


def get_experiment_name(cfg):
    experiment_name = f'{cfg.ID}' \
                      f'_{cfg.OPTION}' \
                      f'{"_pretr" if cfg.PRETRAINED else ""}{"Copy" if cfg.PRETRAINED and cfg.COPY_RGB_WEIGHTS else ""}' \
                      f'_{cfg.TARGET_SIZE[0]}' \
                      f'{"_Aug-" if len(cfg.AUGMENTATIONS) > 0 else ""}{"-".join(cfg.AUGMENTATIONS)}' \
                      f'{"_strat" if cfg.STRATIFIED else ""}' \
                      f'_lr{cfg.LR}' \
                      f'_b{cfg.BATCH_SIZE}'
    print("Experiment name:")
    print(experiment_name)
    return experiment_name


def find_last_values_tensorboard(log_dir, tag):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    assert len(event_files) > 0, "No event files found in log directory."
    event_files.sort(key=os.path.getmtime)

    event_file = event_files[-1]  # Get the latest event file.
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_events = event_acc.Scalars(tag)
    assert len(scalar_events) > 0, f"No events found for tag '{tag}' in {log_dir}."
    return scalar_events[-1].value

#
# def imshow_keypoints(img,
#                      pose_result,
#                      skeleton=None,
#                      kpt_score_thr=0.3,
#                      pose_kpt_color=None,
#                      pose_link_color=None,
#                      radius=4,
#                      thickness=1,
#                      show_keypoint_weight=False):
#     """Draw keypoints and links on an image.
#
#     Args:
#             img (str or Tensor): The image to draw poses on. If an image array
#                 is given, id will be modified in-place.
#             pose_result (list[kpts]): The poses to draw. Each element kpts is
#                 a set of K keypoints as an Kx3 numpy.ndarray, where each
#                 keypoint is represented as x, y, score.
#             kpt_score_thr (float, optional): Minimum score of keypoints
#                 to be shown. Default: 0.3.
#             pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
#                 the keypoint will not be drawn.
#             pose_link_color (np.array[Mx3]): Color of M links. If None, the
#                 links will not be drawn.
#             thickness (int): Thickness of lines.
#     """
#
#     img_h, img_w, _ = img.shape
#
#     for kpts in pose_result:
#
#         kpts = np.array(kpts, copy=False)
#
#         # draw each point on image
#         if pose_kpt_color is not None:
#             assert len(pose_kpt_color) == len(kpts)
#
#             for kid, kpt in enumerate(kpts):
#                 x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
#
#                 if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
#                     # skip the point that should not be drawn
#                     continue
#
#                 color = tuple(int(c) for c in pose_kpt_color[kid])
#                 if show_keypoint_weight:
#                     img_copy = img.copy()
#                     cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
#                                color, -1)
#                     transparency = max(0, min(1, kpt_score))
#                     cv2.addWeighted(
#                         img_copy,
#                         transparency,
#                         img,
#                         1 - transparency,
#                         0,
#                         dst=img)
#                 else:
#                     cv2.circle(img, (int(x_coord), int(y_coord)), radius,
#                                color, -1)
#
#         # draw links
#         if skeleton is not None and pose_link_color is not None:
#             assert len(pose_link_color) == len(skeleton)
#
#             for sk_id, sk in enumerate(skeleton):
#                 pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
#                 pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
#
#                 if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
#                         or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
#                         or pos2[1] <= 0 or pos2[1] >= img_h
#                         or kpts[sk[0], 2] < kpt_score_thr
#                         or kpts[sk[1], 2] < kpt_score_thr
#                         or pose_link_color[sk_id] is None):
#                     # skip the link that should not be drawn
#                     continue
#                 color = tuple(int(c) for c in pose_link_color[sk_id])
#                 if show_keypoint_weight:
#                     img_copy = img.copy()
#                     X = (pos1[0], pos2[0])
#                     Y = (pos1[1], pos2[1])
#                     mX = np.mean(X)
#                     mY = np.mean(Y)
#                     length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
#                     angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
#                     stickwidth = 2
#                     polygon = cv2.ellipse2Poly(
#                         (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
#                         int(angle), 0, 360, 1)
#                     cv2.fillConvexPoly(img_copy, polygon, color)
#                     transparency = max(
#                         0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
#                     cv2.addWeighted(
#                         img_copy,
#                         transparency,
#                         img,
#                         1 - transparency,
#                         0,
#                         dst=img)
#                 else:
#                     cv2.line(img, pos1, pos2, color, thickness=thickness)
#
#     return img
