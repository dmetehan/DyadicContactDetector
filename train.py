import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from argparse import ArgumentParser
from ContactClassifier import ContactClassifier, initialize_model
from utils import parse_config, get_experiment_name, find_last_values_tensorboard

# this is important for FLickrCI3DClassification. It doesn't allow importint v2 after initializing the network.
import torchvision.transforms.v2 as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on {device}")
ContactClassifier(backbone="resnet50")
# net should be initialized here before importing torchvision
from dataset.FlickrCI3DClassification import init_datasets_with_cfg


def train_one_epoch(model, optimizer, loss_fn, train_loader, epoch_index, tb_writer, batch_size):
    overall_loss = 0.
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    i = 0
    all_preds, all_labels = np.zeros((len(train_loader), batch_size)), np.zeros((len(train_loader), batch_size))
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        _, inputs, labels = data
        all_labels[i, :len(labels)] = labels
        inputs = inputs.to(device)
        labels = nn.functional.one_hot(labels, num_classes=2).to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        all_preds[i, :len(labels)] = torch.argmax(outputs.detach().cpu(), dim=1)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        cur_loss = loss.detach().item()
        running_loss += cur_loss
        tb_x = epoch_index * len(train_loader) + i + 1
        tb_writer.add_scalar('Loss/train', cur_loss, tb_x)
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('\tbatch {} loss: {}'.format(i + 1, last_loss))
            overall_loss += running_loss
            running_loss = 0.
    all_preds = all_preds.flatten()[:-(batch_size - len(labels))]
    all_labels = all_labels.flatten()[:-(batch_size - len(labels))]
    acc_blncd = balanced_accuracy_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    if i % 1000 != 999:
        last_loss = running_loss / (i % 1000 + 1)  # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        overall_loss += running_loss
    overall_loss = overall_loss / len(train_loader)
    return overall_loss, acc_blncd, acc, f1


def train_model(model, optimizer, loss_fn, experiment_name, cfg, train_loader, validation_loader,
                exp_dir="experiments", start_epoch=0, resume=False):
    best_model_path = ''
    if resume:
        timestamps = ['_'.join(folder_name.split('_')[-2:]) for folder_name in sorted(os.listdir(exp_dir))
            if (experiment_name in folder_name) and os.path.isdir(os.path.join(exp_dir, folder_name))]
        timestamp = sorted(timestamps)[-1]
        log_dir = f'{exp_dir}/{experiment_name}_{timestamp}/Training vs. Validation Loss_Validation'
        tag = 'Training vs. Validation Loss'  # Replace with your desired tags
        best_vloss = find_last_values_tensorboard(log_dir, tag)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_vloss = 1_000_000.
    writer = SummaryWriter('{}/{}_{}'.format(exp_dir, experiment_name, timestamp))


    model = model.to(device)
    for epoch in range(start_epoch, cfg.EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, acc_blncd, acc, f1 = train_one_epoch(model, optimizer, loss_fn, train_loader, epoch, writer, cfg.BATCH_SIZE)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        i = 0
        all_preds, all_labels = np.zeros((len(validation_loader), cfg.BATCH_SIZE)), np.zeros((len(validation_loader), cfg.BATCH_SIZE))
        for i, vdata in enumerate(validation_loader):
            _, vinputs, vlabels = vdata
            all_labels[i, :len(vlabels)] = vlabels
            vinputs = vinputs.to(device)
            vlabels = nn.functional.one_hot(vlabels, num_classes=2).to(device)
            voutputs = model(vinputs)
            all_preds[i, :len(vlabels)] = torch.argmax(voutputs.detach().cpu(), dim=1)
            vloss = loss_fn(voutputs, vlabels.float())
            running_vloss += vloss.detach()
        all_preds = all_preds.flatten()[:-(cfg.BATCH_SIZE - len(vlabels))]
        all_labels = all_labels.flatten()[:-(cfg.BATCH_SIZE - len(vlabels))]
        vacc_blncd = balanced_accuracy_score(all_labels, all_preds)
        vacc = accuracy_score(all_labels, all_preds)
        vf1 = f1_score(all_labels, all_preds)
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {} - ACC_BLNCD train {} valid {}'.format(avg_loss, avg_vloss, acc_blncd, vacc_blncd))

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.add_scalars('Training vs. Validation Balanced Accuracy',
                           {'Training': acc_blncd, 'Validation': vacc_blncd},
                           epoch + 1)
        writer.add_scalars('Training vs. Validation Balanced Accuracy',
                           {'Training': acc, 'Validation': vacc},
                           epoch + 1)
        writer.add_scalars('Training vs. Validation F1 Score',
                           {'Training': f1, 'Validation': vf1},
                           epoch + 1)
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}/{}_{}_{}'.format(exp_dir, experiment_name, timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path

    print('Finished Training')
    print(f'Best model is saved at: {best_model_path}')
    return best_model_path


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    parser.add_argument('--resume', action='store_true', default=False, help='False: start from scratch, '
                                                                               'True: continue the last experiment')
    exp_dir = "experiments"
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)

    # train_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/train'
    # test_dir_hdd = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D Classification/test'
    train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    model, optimizer, loss_fn = initialize_model(cfg, device)
    experiment_name = get_experiment_name(cfg)
    start_epoch = 0
    if args.resume:
        models = [(file_name, int(file_name.split('_')[-1])) for file_name in sorted(os.listdir(exp_dir))
                              if (experiment_name in file_name) and os.path.isfile(os.path.join(exp_dir, file_name))]
        models.sort(key=lambda x: x[1])
        model_name, start_epoch = models[-1]
        model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(train_dir_ssd, test_dir_ssd, cfg)
    best_model_path = train_model(model, optimizer, loss_fn, experiment_name, cfg, train_loader, validation_loader,
                                  exp_dir=exp_dir, start_epoch=start_epoch, resume=args.resume)
    # TODO: Write best model's name/path to a file after the training is completed.


if __name__ == '__main__':
    main()
