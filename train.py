import os
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from argparse import ArgumentParser
from ContactClassifier import ContactClassifier
from utils import parse_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on {device}")
ContactClassifier(backbone="resnet50")
# net should be initialized here before importing torchvision
from dataset.FlickrCI3DClassification import init_datasets_with_cfg


def train_one_epoch(model, optimizer, loss_fn, train_loader, epoch_index, tb_writer):
    overall_loss = 0.
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    i = 0
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


def train_model(model, optimizer, loss_fn, experiment_name, cfg, train_loader, validation_loader):
    exp_folder = "experiments"
    best_model_path = ''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('{}/{}_{}'.format(exp_folder, experiment_name, timestamp))

    best_vloss = 1_000_000.

    model = model.to(device)
    for epoch in range(cfg.EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, epoch, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        i = 0
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
                           epoch + 1)
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}/{}_{}_{}'.format(exp_folder, experiment_name, timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path

    print('Finished Training')
    print(f'Best model is saved at: {best_model_path}')
    return best_model_path


def initialize_model(cfg):
    model = ContactClassifier(backbone="resnet50", weights="IMAGENET1K_V2" if cfg.PRETRAINED else None,
                              option=cfg.OPTION,
                              copy_rgb_weights=cfg.COPY_RGB_WEIGHTS)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(cfg.LOSS_WEIGHTS).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    return model, optimizer, loss_fn


def set_experiment_name(cfg):
    experiment_name = f'{cfg.ID}' \
                      f'_{cfg.OPTION}' \
                      f'{"_pretr" if cfg.PRETRAINED else ""}{"Copy" if cfg.PRETRAINED and cfg.COPY_RGB_WEIGHTS else ""}' \
                      f'_{cfg.TARGET_SIZE[0]}' \
                      f'{"_Aug-" if len(cfg.AUGMENTATIONS) > 0 else ""}{"-".join(cfg.AUGMENTATIONS)}' \
                      f'{"_strat" if cfg.STRATIFIED else ""}' \
                      f'_lr{cfg.LR}' \
                      f'_b{cfg.BATCH_SIZE}'
    print("Experiment name:")
    print(experiment_name)
    return experiment_name


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)

    # train_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train'
    # test_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test'
    train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    model, optimizer, loss_fn = initialize_model(cfg)
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(train_dir_ssd, test_dir_ssd, cfg)
    experiment_name = set_experiment_name(cfg)
    best_model_path = train_model(model, optimizer, loss_fn, experiment_name, cfg, train_loader, validation_loader)
    # TODO: Write best model's name/path to a file after the training is completed.


if __name__ == '__main__':
    main()
