import os
import cv2
import numpy as np
from argparse import ArgumentParser

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

from dataset.YOUth10mClassification import YOUth10mClassification

CLASSES = ("no touch", "touch")

def get_predictions(model, model_name, dataset, device):
    model.eval()
    model = model.to(device)
    print(f'Testing on test set using {model_name}')
    y_pred = []
    y_true = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    output = ''
    with torch.no_grad():
        for i in range(len(dataset)):
            idx, images, labels = dataset[i]
            if torch.cuda.is_available():
                images = torch.from_numpy(np.expand_dims(images, axis=0)).to(device)
                labels = torch.from_numpy(np.array([labels])).to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            pred = predicted.cpu().item()
            y_pred.append(pred)  # Save Prediction

            label = labels.data.cpu().numpy()[0]
            y_true.append(label)  # Save Truth
            if pred != label:
                crop_path = dataset.img_labels_dets.loc[i, "crop_path"]
                subject, frame_no = crop_path.split('.')[0].split('/')[-2:]
                print(subject, frame_no, pred, label)
                output += f"{crop_path}, {label}\n"

    acc = accuracy_score(y_true, y_pred)
    acc_blncd = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'Accuracy of the network on the test images ({len(y_true)}): {acc:.4f}')
    print(f'Balanced accuracy of the network on the test images ({len(y_true)}): {acc_blncd:.4f}')
    print(f'F1 Score of the network on the test images ({len(y_true)}): {f1:.4f}')
    with open(f'incorrect_predictions_{model_name.split("_")[0]}.txt', 'w') as f:
        f.write(output)
    return y_pred, y_true


def load_model_weights(model, model_name, exp_dir):
    model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('model_name', help='model name to be used')

    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)
    root_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all'
    model, _, _ = initialize_model(cfg, device)
    cfg.BATCH_SIZE = 1 # to get accurate results
    test_dataset = YOUth10mClassification(root_dir_ssd, option=cfg.OPTION, target_size=cfg.TARGET_SIZE, augment=cfg.AUGMENTATIONS,
                                          bodyparts_dir=cfg.BODYPARTS_DIR, _set='test')
    load_model_weights(model, args.model_name, args.exp_dir)
    y_pred, y_true = get_predictions(model, args.model_name, test_dataset, device)
    print(y_pred)
    print(y_true)


if __name__ == '__main__':
    main()
