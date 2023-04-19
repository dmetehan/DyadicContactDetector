from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn

from ContactClassifier import ContactClassifier

ContactClassifier(backbone="resnet50")
# net should be initialized here before importing torchvision
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from dataset.FlickrCI3DClassification import init_datasets_with_cfg_dict


def train_flickr(cfg, checkpoint_dir=None, train_dir=None, test_dir=None):
    model = ContactClassifier(backbone="resnet50", weights="IMAGENET1K_V2" if cfg["PRETRAINED"] else None,
                              option=cfg["OPTION"],
                              copy_rgb_weights=cfg["COPY_RGB_WEIGHTS"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    print(f"Working on {device}")
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(cfg["LOSS_WEIGHTS"]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader, val_loader, _ = init_datasets_with_cfg_dict(train_dir, test_dir, cfg)


    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            _, inputs, labels = data
            inputs, labels = inputs.to(device), nn.functional.one_hot(labels, num_classes=2).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.detach().item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                _, inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels_one_hot = nn.functional.one_hot(labels, num_classes=2)

                outputs = model(inputs)
                predicted = torch.argmax(outputs.detach(), dim=1)
                total += labels.size(0)

                tp += ((predicted==1) & (labels==1)).sum().item()
                fp += ((predicted==1) & (labels==0)).sum().item()
                tn += ((predicted==0) & (labels==0)).sum().item()
                fn += ((predicted==0) & (labels==1)).sum().item()
                correct += (predicted == labels).sum().item()

                loss = loss_fn(outputs, labels_one_hot.float())
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        sensitivity = recall
        specificity = tn / (fp + tn)
        tune.report(loss=(val_loss / val_steps),
                    accuracy=correct / total,
                    balanced_accuracy=(sensitivity+specificity)/2,
                    f1_score=2*precision*recall/(precision+recall))
    print("Finished Training")


def test_accuracy(model, cfg, device="cpu", train_dir=None, test_dir=None):
    _, _, test_loader = init_datasets_with_cfg_dict(train_dir, test_dir, cfg)

    correct = 0
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with torch.no_grad():
        for data in test_loader:
            _, inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.detach(), dim=1)
            total += labels.size(0)
            tp += ((predicted==1) & (labels==1)).sum().item()
            fp += ((predicted==1) & (labels==0)).sum().item()
            tn += ((predicted==0) & (labels==0)).sum().item()
            fn += ((predicted==0) & (labels==1)).sum().item()
            correct += (predicted == labels).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = recall
    specificity = tn / (fp + tn)
    accuracy = correct / total
    balanced_accuracy = (sensitivity+specificity)/2
    f1_score = 2*precision*recall/(precision+recall)
    return accuracy, balanced_accuracy, f1_score


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    config = {
        "OPTION": "jointmaps_rgb_bodyparts",
        "PRETRAINED": tune.choice([True, False]),
        "COPY_RGB_WEIGHTS": tune.choice([True, False]),
        "TARGET_SIZE": tune.choice([[112, 112], [224, 224], [56, 56]]),
        "AUGMENTATIONS": tune.choice([["hflip"], ["crop"], ["swap"], ["hflip", "crop"], ["hflip", "swap"],
                                      ["crop", "swap"], ["hflip", "crop", "swap"]]),
        "STRATIFIED": True,
        "LOSS_WEIGHTS": [1, 1],
        "LR": tune.loguniform(1e-4, 1e-1),
        "BATCH_SIZE": tune.choice([2, 4, 8, 16, 32, 64]),
        "BODYPARTS_DIR": "bodyparts_binary"
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "balanced_accuracy", "f1_score", "training_iteration"])
    result = tune.run(
        partial(train_flickr, train_dir=train_dir_ssd, test_dir=test_dir_ssd),
        name="train_flickr_2023-04-06_16-18-42",
        local_dir="/home/sac/ray_results",
        resume=True,
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = ContactClassifier(backbone="resnet50", weights="IMAGENET1K_V2" if config["PRETRAINED"] else None,
                              option=best_trial.config["OPTION"],
                              copy_rgb_weights=best_trial.config["COPY_RGB_WEIGHTS"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc, test_acc_blncd, test_f1_score = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}\tbalanced_accuracy: {}\tf1_score:{}".format(test_acc, test_acc_blncd, test_f1_score))


def test_functions():
    train_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
    test_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
    config = {
        "OPTION": "jointmaps_rgb_bodyparts",
        "PRETRAINED": True,
        "COPY_RGB_WEIGHTS": True,
        "TARGET_SIZE": [56, 56],
        "AUGMENTATIONS": [],
        "STRATIFIED": True,
        "LOSS_WEIGHTS": [1, 1],
        "LR": 1e-4,
        "BATCH_SIZE": 2,
        "BODYPARTS_DIR": "bodyparts_binary"
    }
    train_flickr(config, train_dir=train_dir_ssd, test_dir=test_dir_ssd)


if __name__ == "__main__":
    # test_functions()
    # You can change the number of GPUs per trial here:
    main(num_samples=25, max_num_epochs=10, gpus_per_trial=2)