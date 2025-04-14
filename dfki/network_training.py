import os
import torch
import numpy as np
import torchmetrics
from pathlib import Path
from typing import Optional
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from dfki.data import DetectedRobotImages
from dfki.network_definitions import CNN3D, CNNLSTM, PretrainedLinear, RobotNet


def run_dataloader(
    network,
    dataloader,
    epoch,
    num_epochs,
    optimizer,
    loss_function,
    metrics,
    train,
    device,
):
    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("loss: {task.fields[loss]:.4f}"),
        *[
            TextColumn("{}: {{task.fields[{}]:.2f}}".format(metric_name, metric_name))
            for metric_name in metrics
        ],
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("(test)" if not train else ""),
    )
    epoch_task = progress_bar.add_task(
        "epoch: [{}/{}]".format(epoch + 1, num_epochs), total=len(dataloader)
    )
    progress_bar.update(
        task_id=epoch_task,
        advance=0,
        **{metric: 0 for metric in metrics},  # type: ignore
        loss=0,
    )

    network.train() if train else network.eval()

    with progress_bar as progress:
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)

            match loss_function:
                case nn.CrossEntropyLoss():
                    loss = loss_function(outputs, labels)
                case nn.BCELoss():
                    loss = loss_function(outputs, labels)
                case nn.NLLLoss():
                    # this expects 1D labels, not one-hot
                    if labels[0].dim() != 0:
                        labels = labels.argmax(dim=1)
                    loss = loss_function(torch.log(outputs), labels)
                case nn.BCEWithLogitsLoss():
                    loss = loss_function(outputs, labels)
                    outputs = outputs.sigmoid()
                case _:
                    raise ValueError("The loss function should be one of the cases")

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                sequence_metrics = {
                    metric_name: (
                        (idx * progress.tasks[epoch_task].fields[metric_name])
                        + metric(outputs, labels).item()
                    )
                    / (idx + 1)
                    for metric_name, metric in metrics.items()
                }

            progress.update(
                task_id=epoch_task,
                advance=1,
                **sequence_metrics,
                loss=((idx * progress.tasks[epoch_task].fields["loss"]) + loss.item())
                / (idx + 1),
            )

    return network, sequence_metrics


def cross_validation(
    video_indices: list[int], num_test_vids: int, num_folds: int, seed: Optional[int]
):

    if seed is not None:
        np.random.seed(seed)

    test_videos = np.random.choice(video_indices, num_test_vids)

    remaining_videos = np.setdiff1d(video_indices, test_videos)
    np.random.shuffle(remaining_videos)

    val_splits = np.array_split(remaining_videos, num_folds)

    splits = []
    for val_videos in val_splits:
        train_videos = np.setdiff1d(remaining_videos, val_videos)
        splits.append({"train": train_videos.tolist(), "val": val_videos.tolist()})

    return test_videos, splits


import copy


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path=None):
        """
        Args:
            patience (int): How many epochs to wait without improvement.
            min_delta (float): Minimum improvement in F1 to reset patience.
            save_path (str): If set, best model is saved to this path.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")  # higher is better for F1
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None
        self.save_path = save_path

    def __call__(self, val_f1, model):
        if val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            if self.save_path:
                torch.save(self.best_model_wts, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":

    # declare datasets variables
    downsample_img_by = 8
    downsample_sequence = True
    imgs_per_sec = 1
    image_sequences = False
    imgs_per_sequence = 5
    time_spacing = 1.0
    dataset_root = Path(__file__).parents[4] / "srv/evenflow-data/DFKI/Dataset_3"

    # get train/val/test splits
    test_videos, splits = cross_validation(
        video_indices=list(range(10)),
        num_test_vids=0,
        num_folds=5,
        seed=42,
    )

    results_per_split = {}
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    model_save_dir = Path(__file__).parent / f"saved_models/{now}"
    os.makedirs(model_save_dir)

    # iterate through all train/validation splits
    for i, inner in enumerate(splits):
        print(f"Split {i+1}/{len(splits)}")

        # create train/val datasets
        train_dataset, val_dataset = [
            DetectedRobotImages(
                downsample_img_by,
                downsample_sequence,
                imgs_per_sec,
                image_sequences,
                imgs_per_sequence,
                time_spacing,
                idxs,
                dataset_root,
            )
            for idxs in [inner["train"], inner["val"]]
        ]

        # create CNN
        net = PretrainedLinear(num_classes=len(train_dataset[0][1]))

        # define training config
        lr = 1e-3
        batch_size = 32
        num_epochs = 20
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(net.parameters(), lr=lr)
        loss_function = nn.NLLLoss()

        # create dataloaders for training and validation
        train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size, shuffle=True)

        # define training metrics
        metrics = {
            "f1-macro": torchmetrics.F1Score(
                task="multiclass",
                average="macro",
                num_classes=len(train_dataset[0][1]),
            ).to(device),
            "f1-micro": torchmetrics.F1Score(
                task="multiclass",
                average="micro",
                num_classes=len(train_dataset[0][1]),
            ).to(device),
            "accuracy": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=len(train_dataset[0][1]),
            ).to(device),
        }

        # instantiate early stopping class
        # this will track validation macro-F1 and save the best-performing model
        early_stopper = EarlyStopping(
            patience=5,
            min_delta=0.01,
            save_path=model_save_dir / f"{net.__class__.__name__}_split_{i+1}of{len(splits)}.pt",
        )

        net.to(device)
        for epoch in range(num_epochs):
            net, train_metrics = run_dataloader(
                net,
                train_dl,
                epoch,
                num_epochs,
                optimizer,
                loss_function,
                metrics,
                train=True,
                device=device,
            )

            net, val_metrics = run_dataloader(
                net,
                val_dl,
                epoch,
                num_epochs,
                optimizer,
                loss_function,
                metrics,
                train=False,
                device=device,
            )

            early_stopper(val_metrics["f1-macro"], net)

            if early_stopper.early_stop or epoch + 1 == num_epochs:
                results_per_split[i + 1] = {
                    "model": net,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }

    for fold, data in results_per_split.items():
        print(f"Fold {fold}:")
        train_metrics = data["train"]
        val_metrics = data["val"]
        
        print("  Train:", end=" ")
        print(" | ".join(f"{k}: {v:.3f}" for k, v in train_metrics.items()))
        
        print("  Val:  ", end=" ")
        print(" | ".join(f"{k}: {v:.3f}" for k, v in val_metrics.items()))
        
        print()
    