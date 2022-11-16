import os
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

from ContactClassifier import init_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
OPTION = 3
model = init_model(option=OPTION)
model = model.to(device)
# net should be initialized here before importing torchvision
from dataset.FlickrCI3DClassification import init_datasets


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
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
            running_loss = 0.
    last_loss = running_loss / (i % 1000 + 1)  # loss per batch
    print('  batch {} loss: {}'.format(i + 1, last_loss))
    tb_x = epoch_index * len(train_loader) + i + 1
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    return last_loss


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

batch_size = 32
# train_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train'
# test_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test'
train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/train'
test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/test'
train_loader, validation_loader, test_loader = init_datasets(train_dir_ssd, test_dir_ssd, batch_size, option=OPTION,
                                                             target_size=(112, 112))
experiment_name = 'gauss_hmaps_rgb_pretrained_strat_weighted_112'

# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/{}_{}'.format(experiment_name, timestamp))
epoch_number = 0

EPOCHS = 10

best_vloss = 1_000_000.

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
        if torch.cuda.is_available():
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
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
