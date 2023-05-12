import os
from argparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score
import seaborn as sn
import pandas as pd
from tqdm import tqdm

from ContactClassifier import ContactClassifier, initialize_model
from utils import parse_config, get_experiment_name

# this is important for FLickrCI3DClassification. It doesn't allow importint v2 after initializing the network.
import torchvision.transforms.v2 as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
_ = ContactClassifier()
# net should be initialized here before importing torchvision

from dataset.YOUth10mClassification import YOUth10mClassification, SET_SPLITS

CLASSES = ("no touch", "touch")
import json


def test_model(model, model_name, exp_name, exp_dir, dataset, device):
    model.eval()
    model = model.to(device)
    print(f'Testing on test set using {model_name}')
    y_pred = []
    y_true = []
    video_pred_dict = {}
    for subject in SET_SPLITS['test']:
        video_pred_dict[subject] = {'pred': [], 'true': []}
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i in range(len(dataset)):
            print(dataset.img_labels_dets.loc[i, "crop_path"])
            _, images, labels = dataset[i]
            if torch.cuda.is_available():
                images = torch.from_numpy(np.expand_dims(images, axis=0)).to(device)
                labels = torch.from_numpy(np.array([labels])).to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth
            for subject in video_pred_dict:
                if subject in dataset.img_labels_dets.loc[i, "crop_path"]:
                    video_pred_dict[subject]['pred'].extend(predicted.cpu().numpy())
                    video_pred_dict[subject]['true'].extend(labels)
    print(video_pred_dict)
    acc = accuracy_score(y_true, y_pred)
    acc_blncd = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy of the network on the test images ({len(y_true)}): {acc:.4f}')
    print(f'Balanced accuracy of the network on the test images ({len(y_true)}): {acc_blncd:.4f}')
    print(f'F1 Score of the network on the test images ({len(y_true)}): {f1:.4f}')
    with open("video_pred_dict.txt", "w") as fp:
        json.dump(video_pred_dict, fp)  # encode dict into JSON


def load_model_weights(model, exp_dir, exp_name):
    models = [(file_name, int(file_name.split('_')[-1])) for file_name in sorted(os.listdir(exp_dir))
              if (exp_name in file_name) and os.path.isfile(os.path.join(exp_dir, file_name))]
    models.sort(key=lambda x: x[1])
    model_name = models[-1][0]
    model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))
    return model_name


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('exp_name', help='experiment name')
    parser.add_argument('--test_set', action='store_true', default=False, help='False: validation set, True: test set')
    parser.add_argument('--finetune', action='store_true', default=False, help='False: no finetuning'
                                                                               'True: finetune on YOUth')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)
    root_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all'
    model, _, _ = initialize_model(cfg, device)
    if args.finetune:
        cfg.LR = cfg.LR / 10
    exp_name = args.exp_name
    if args.finetune:
        exp_name = f'finetune_{exp_name}'
    cfg.BATCH_SIZE = 1 # to get accurate results
    test_dataset = YOUth10mClassification(root_dir_ssd, option=cfg.OPTION, target_size=cfg.TARGET_SIZE, augment=cfg.AUGMENTATIONS,
                                          bodyparts_dir=cfg.BODYPARTS_DIR, _set='test')
    model_name = load_model_weights(model, args.exp_dir, exp_name)
    test_model(model, model_name, exp_name, args.exp_dir, test_dataset, device)


if __name__ == '__main__':
    main()
