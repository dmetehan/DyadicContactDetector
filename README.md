# Embracing Contact: Detecting Parent-Infant Interactions
## Metehan Doyran, Ronald Poppe, Albert Ali Salah

## Dataset:
* [https://www.uu.nl/en/research/youth-cohort-study/request-youth-data](YOUth Parent-Child Interaction Dataset)

## Requirements:
* mmpose (git submodule update --init --recursive --remote)
* Python 3.8 (virtualenv --python=/usr/bin/python3.8 venv && source venv/bin/activate && pip install -r requirements.txt)

## To run:
1. Crop images around interacting people:<br>
<code>python prep_crops.py '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification'</code>
2. Create a new environment to run mmdet and mmpose on the crops:<br>
<code>sh create_conda_env_for_mmpose.sh</code>
3. Run the pose detector on the train set using the new conda environment:<br>
<code>~/anaconda3/envs/openmmlab/bin/python top_down_img_inference_with_mmdet.py mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth --img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/train/crops" --out-img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/train/vis_results"</code>
4. Run the pose detector on the test set using the new conda environment:<br>
<code>~/anaconda3/envs/openmmlab/bin/python top_down_img_inference_with_mmdet.py mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco.py mmpose-utils/mmdet_yolo/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth mmpose-utils/hrnet_w48_comb_R0_384x288_dark.py mmpose-utils/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth --img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/test/crops" --out-img-root "/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/test/vis_results"</code>
5. Getting the bodypart segmentation clone and install the following repository: https://github.com/kevinlin311tw/CDCL-human-part-segmentation.git
6. sudo docker run --runtime=nvidia -v /mnt/hdd1/GithubRepos/CDCL-human-part-segmentation:/workspace -v /mnt/hdd1/Datasets/CI3D/FlickrCI3D-Classification/train:/train -it cdcl:v1 bash
7. python3 inference_15parts.py --scale=1 --scale=0.5 --scale=0.75 --input_folder /train/crops/ --output_folder /train/bodyparts_binary/
8. 
