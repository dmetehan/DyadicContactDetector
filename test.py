import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd
from ContactClassifier import init_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
OPTION = 3
model = init_model(option=OPTION)
model.load_state_dict(torch.load("runs/gauss_hmaps_rgb_pretrained_strat_weighted_112_20221101_135239_3"))
model.eval()
model = model.to(device)
# net should be initialized here before importing torchvision

from dataset.FlickrCI3DClassification import init_datasets

batch_size = 32
train_dir = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train'
test_dir = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test'
train_loader, validation_loader, test_loader = init_datasets(train_dir, test_dir, batch_size, option=OPTION)

classes = ("no touch", "touch")

y_pred = []
y_true = []
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
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
plt.savefig('output.png')
print(f'Accuracy of the network on the test images: {accuracy_score(y_true, y_pred):.3f}')
