import os
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

from ContactClassifier import ContactClassifier
from dataset.utils import Aug, Options
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
PRETRAINED = True
OPTION = Options.jointmaps_rgb_bodyparts
TARGET_SIZE = (112, 112)
DROPOUT = (None, None)
COPY_RGB_WEIGHTS = True
BODYPARTS_DIR = "bodyparts_binary"
model = ContactClassifier(backbone="resnet50", weights="IMAGENET1K_V2" if PRETRAINED else None,
                          option=OPTION,
                          dropout=DROPOUT,
                          copy_rgb_weights=COPY_RGB_WEIGHTS)
# net should be initialized here before importing torchvision
from dataset.FlickrCI3DClassification import init_datasets


def train_one_epoch(epoch_index, tb_writer):
    overall_loss = 0.
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = nn.functional.one_hot(labels, num_classes=2).to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.detach().item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            overall_loss += running_loss
            running_loss = 0.
    if i % 1000 != 999:
        overall_loss += running_loss
        last_loss = running_loss / (i % 1000 + 1)  # loss per batch
    print('  batch {} loss: {}'.format(i + 1, last_loss))
    overall_loss = overall_loss / len(train_loader)
    tb_x = epoch_index * len(train_loader) + i + 1
    tb_writer.add_scalar('Loss/train', overall_loss, tb_x)
    return overall_loss


# Aug.swap: Swapping the order of pose detections between people (50% chance)
#           NOT WORKING WELL because I believe the order is always top left to bottom right and switching it randomly doesn't help
# Aug.hflip: Horizontally flipping the rgb image as well as flipping left/right joints
# Aug.crop: Cropping
AUGMENTATIONS = (Aug.hflip, Aug.crop)
loss_weights = [1, 3.35]
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(loss_weights).to(device))
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# TODO: Write all the parameters into config files per experiment.
batch_size = 32
# train_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train'
# test_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test'
train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
train_loader, validation_loader, test_loader = init_datasets(train_dir_ssd, test_dir_ssd, batch_size, option=OPTION,
                                                             target_size=TARGET_SIZE, augment=AUGMENTATIONS, bodyparts_dir=BODYPARTS_DIR)
experiment_name = f'{"joint_hmaps" if OPTION in [Options.jointmaps, Options.jointmaps_rgb, Options.jointmaps_rgb_bodyparts] else "gauss_hmaps"}' \
                  f'{"_rgb" if OPTION in [Options.gaussian_rgb, Options.jointmaps_rgb, Options.gaussian_rgb_bodyparts, Options.jointmaps_rgb_bodyparts] else ""}' \
                  f'{"_pretrained" if PRETRAINED else ""}{"Copied" if PRETRAINED and COPY_RGB_WEIGHTS else ""}' \
                  f'{"_bodyparts" if BODYPARTS_DIR else ""}' \
                  f'_strat' \
                  f'_{TARGET_SIZE[0]}' \
                  f'{f"_first{DROPOUT[0]}" if DROPOUT[0] else ""}' \
                  f'{f"_last{DROPOUT[1]}" if DROPOUT[1] else ""}' \
                  f'_reproduction' \
                  f'_BCEWithLogitsLoss' \
                  f'_AugCorrect_{"_".join(AUGMENTATIONS)}'
print(experiment_name)
# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/{}_{}'.format(experiment_name, timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

model = model.to(device)
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.to(device)
        vlabels = nn.functional.one_hot(vlabels, num_classes=2).to(device)
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels.float())
        running_vloss += vloss.detach()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track the best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'runs/{}_{}_{}'.format(experiment_name, timestamp, epoch_number + 1)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

print('Finished Training')
