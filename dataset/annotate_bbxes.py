import os
from json import JSONDecodeError

import cv2
import json
import numpy as np
import pandas as pd


def calc_area(x1, y1, x2, y2, conf):
    return (x2 - x1) * (y2 - y1)


def assign_colors(bbxes):
    if len(bbxes) == 0:
        return []
    elif len(bbxes) == 1:
        return [(255, 0, 0)]
    else:
        a0, a1 = calc_area(*bbxes[0]), calc_area(*bbxes[1])
        if a0 > a1:
            return [(255, 0, 0), (0, 0, 255)]
        else:
            return [(0, 0, 255), (255, 0, 0)]


def draw_skeleton(frame, pose, bbxes):
    assert len(pose) == len(bbxes) <= 2, f"{len(pose)} {len(bbxes)}"
    colors = assign_colors(bbxes)
    for p in range(len(bbxes)):
        cv2.rectangle(frame, list(map(round, bbxes[p][:2])), list(map(round, bbxes[p][2:4])), colors[p])
    for p in range(len(pose)):
        for k in range(17):
            x, y, _ = pose[p][k]
            cv2.circle(frame, (round(x), round(y)), 5, colors[p], -1)


def main():
    dets_file = "/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all/pose_detections.json"
    pose_dets = json.load(open(dets_file))
    for idx, label in pose_dets['contact_type'].items():
        if label == '1':
            continue  # skip ambiguous frames
        img, window_name = cv2.imread(pose_dets['crop_path'][idx]), "_".join(pose_dets['crop_path'][idx].split('/')[-2:])
        draw_skeleton(img, pose_dets['preds'][idx], pose_dets['bbxes'][idx])
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


if __name__ == '__main__':
    main()
