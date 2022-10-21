# Create a conda environment for mmpose following instructions on https://github.com/open-mmlab/mmpose
# Use that conda environment to detect the poses
~/anaconda3/envs/openmmlab/bin/python \
/mnt/hdd1/GithubRepos/ContactClassification/mmpose-utils/top_down_img_inference_with_mmdet.py \
mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py \
mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py \
mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth \
--img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train/crops" \
--out-img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train/vis_results"

~/anaconda3/envs/openmmlab/bin/python \
/mnt/hdd1/GithubRepos/ContactClassification/mmpose-utils/top_down_img_inference_with_mmdet.py \
mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py \
mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py \
mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth \
--img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test/crops" \
--out-img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test/vis_results"