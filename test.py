import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd
from ContactClassifier import ContactClassifier
from utils import Options
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
OPTION = Options.jointmaps_rgb_bodyparts
TEST_SET = True
BODYPARTS_DIR = "bodyparts_binary"
model = ContactClassifier(option=OPTION)
model.eval()
model = model.to(device)
# net should be initialized here before importing torchvision

from dataset.FlickrCI3DClassification import init_datasets
# from dataset.YOUth10mClassification import init_datasets

batch_size = 32
train_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
# train_dir = ''
test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
# test_dir = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/test'
classes = ("no touch", "touch")

all_models = ['joint_hmaps_rgb_pretrainedCopied_bodyparts_strat_112_reproduction_BCEWithLogitsLoss_AugCorrect_hflip_crop_20230320_153911_2']
for model_name in all_models:
    if 'bodyparts' not in model_name:
        if 'gauss' in model_name:
            if 'rgb' in model_name:
                OPTION = Options.gaussian_rgb
            else:
                OPTION = Options.gaussian
        else:
            if 'rgb' in model_name:
                OPTION = Options.jointmaps_rgb
            else:
                OPTION = Options.jointmaps
    else:
        if 'gauss' in model_name:
            if 'rgb' in model_name:
                OPTION = Options.gaussian_rgb_bodyparts
            else:
                raise NotImplementedError
                # OPTION = Options.gaussian_bodyparts
        else:
            if 'rgb' in model_name:
                OPTION = Options.jointmaps_rgb_bodyparts
            else:
                raise NotImplementedError
                # OPTION = Options.jointmaps_bodyparts

    TARGET_SIZE = (224, 224)
    if '_112_' in model_name:
        TARGET_SIZE = (112, 112)
    DROPOUT = [None, None]
    if 'first' in model_name:
        for element in model_name.split('_'):
            if 'first' in element:
                DROPOUT[0] = float(element[5:])
    if 'last' in model_name:
        for element in model_name.split('_'):
            if 'last' in element:
                DROPOUT[1] = float(element[4:])
    model = ContactClassifier(backbone="resnet50", weights="IMAGENET1K_V2", option=OPTION)
    model.load_state_dict(torch.load(f"runs/{model_name}"))
    model.eval()
    model = model.to(device)
    train_loader, validation_loader, test_loader = init_datasets(train_dir, test_dir, batch_size, option=OPTION,
                                                                 target_size=TARGET_SIZE, bodyparts_dir=BODYPARTS_DIR)
    data_loader = test_loader if TEST_SET else validation_loader
    print(f'Testing on {"test set" if TEST_SET else "validation set"} using {model_name}')
    y_pred = []
    y_true = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            y_pred.extend(predicted.cpu())  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    acc = accuracy_score(y_true, y_pred)
    plt.savefig(f'confusion_matrices/{"TEST_" if TEST_SET else ""}{f"YOUth_" if "YOUth" in test_dir+train_dir else ""}{model_name}_acc{acc:.3f}.png')
    print(f'Accuracy of the network on the test images ({len(y_true)}): {acc:.3f}')
