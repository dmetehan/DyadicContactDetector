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

from dataset.PublicYOUth10mClassification import init_datasets_with_cfg

classes = ("no touch", "touch")

def get_preds(model, model_path, data_loader, args):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    print(f'Testing on public video using {model_path}')
    y_pred = []
    y_true = []
    softmax_probs = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(data_loader):
            _, images, labels = data
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            softmax_probs.append(torch.nn.functional.softmax(outputs).cpu()[0].tolist())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            y_pred.append(predicted.cpu().item())  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    print(y_pred)
    print(softmax_probs)
    contact = np.count_nonzero(y_pred)
    print(len(y_pred) - contact, contact)
    # acc = accuracy_score(y_true, y_pred)
    # acc_blncd = balanced_accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # print(f'Accuracy of the network on the test images ({len(y_true)}): {acc:.4f}')
    # print(f'Balanced accuracy of the network on the test images ({len(y_true)}): {acc_blncd:.4f}')
    # print(f'F1 Score of the network on the test images ({len(y_true)}): {f1:.4f}')

    # dir_name = [file_name for file_name in sorted(os.listdir(args.exp_dir)) if exp_name in file_name and os.path.isdir(f'{args.exp_dir}/{file_name}')][-1]
    # cf_matrix_norm = confusion_matrix(y_true, y_pred, normalize='true')
    # df_cm = pd.DataFrame(cf_matrix_norm, index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True, cmap='Blues')
    # plt.savefig(f'{args.exp_dir}/{dir_name}/{"TEST" if args.test_set else "VAL"}_YOUth'
    #             f'_acc{100*acc:.2f}_accblncd{100*acc_blncd:.2f}_f1{100*f1:.2f}_norm.png')
    #
    # cf_matrix = confusion_matrix(y_true, y_pred, normalize=None)
    # df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    # plt.savefig(f'{args.exp_dir}/{dir_name}/{"TEST" if args.test_set else "VAL"}_YOUth'
    #             f'_acc{100*acc:.2f}_accblncd{100*acc_blncd:.2f}_f1{100*f1:.2f}.png')


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)
    root_dir = '/mnt/hdd1/Datasets/YentlPublic'
    # model_path = "main/001_jointmaps_rgb_bodyparts_pretrCopy_112_Aug-hflip-crop_strat_lr0.001_b64_20230421_004452_2"
    model_path = "exp/Ablation/a15_jointmaps_bodyparts_pretrCopy_112_Aug-hflip-crop-color_strat_lr0.001_b64_20230424_153247_9"
    model, _, _ = initialize_model(cfg, device)
    cfg.BATCH_SIZE = 1 # to get accurate results
    test_loader = init_datasets_with_cfg(root_dir, root_dir, cfg)
    get_preds(model, model_path, test_loader, args)


if __name__ == '__main__':
    main()
