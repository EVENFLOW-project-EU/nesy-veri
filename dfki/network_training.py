import os
import json
import copy
import torch
import inspect
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
    sequence_metrics = {}

    network.train() if train else network.eval()

    with progress_bar as progress:
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)

            match loss_function:
                case nn.MSELoss():
                    loss = loss_function(outputs, labels)
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
                    loss=(
                        (idx * progress.tasks[epoch_task].fields["loss"]) + loss.item()
                    )
                    / (idx + 1),
                )

    # add the average loss of the epoch to the metrics to be returned
    sequence_metrics["loss"] = progress.tasks[epoch_task].fields["loss"]

    return network, sequence_metrics


def cross_validation(
    video_indices: list[int], num_test_vids: int, num_folds: int, seed: Optional[int]
):

    if seed is not None:
        np.random.seed(seed)

    test_videos = np.random.choice(video_indices, num_test_vids, replace=False)

    remaining_videos = np.setdiff1d(video_indices, test_videos)
    np.random.shuffle(remaining_videos)

    val_splits = np.array_split(remaining_videos, num_folds)

    splits = []
    for val_videos in val_splits:
        train_videos = np.setdiff1d(remaining_videos, val_videos)
        splits.append({"train": train_videos.tolist(), "val": val_videos.tolist()})

    return test_videos, splits



class EarlyStopping:
    def __init__(self, objective, patience=5, min_delta=0.0, save_path=None):
        """
        Args:
            patience (int): How many epochs to wait without improvement.
            min_delta (float): Minimum improvement in tracked metric to reset patience.
            save_path (str): If set, best model is saved to this path.
        """
        assert objective in ["minimize", "maximize"]
        self.objective = objective
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = (
            float("inf") if objective == "minimize" else -float("inf")
        )  # define starting value depending on the type of tracked metric
        self.counter = 0
        self.early_stop = False
        self.best_model_wts = None
        self.save_path = save_path

    def __call__(self, val_metric, model):
        improvement_condition = (
            val_metric < self.best_score - self.min_delta
            if self.objective == "minimize"
            else val_metric > self.best_score + self.min_delta
        )
        if improvement_condition:
            self.best_score = val_metric
            self.counter = 0
            self.best_model_wts = copy.deepcopy(model.state_dict())
            if self.save_path:
                torch.save(self.best_model_wts, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train(net: nn.Module, data_config: dict, train_config: dict, cv_splits: list[dict]):

    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_save_dir = Path(__file__).parent / f"saved_models/{now}"
    os.makedirs(model_save_dir)

    results_per_split = {}
    results_per_split["train_config"] = train_config
    results_per_split["data_config"] = data_config.copy()
    results_per_split["data_config"]["dataset_root"] = str(results_per_split["data_config"]["dataset_root"])

    # iterate through all train/validation splits
    for i, inner in enumerate(cv_splits):
        print(f"Split {i+1}/{len(cv_splits)}")

        # create train/val datasets
        train_dataset, val_dataset = [
            DetectedRobotImages(
                data_config["downsample_img_by"],
                data_config["downsample_sequence"],
                data_config["imgs_per_sec"],
                data_config["image_sequences"],
                data_config["imgs_per_sequence"],
                data_config["time_spacing"],
                data_config["regress"],
                idxs,
                data_config["dataset_root"],
            )
            for idxs in [inner["train"], inner["val"]]
        ]

        # define training config
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(net.parameters(), lr=train_config["learning_rate"])
        loss_function = nn.MSELoss() if data_config["regress"] else nn.NLLLoss()

        # def collate_fn(batch):
        #     batch = list(filter(lambda x: x is not None, batch))
        #     return torch.utils.data.dataloader.default_collate(batch) 
        
        # create dataloaders for training and validation
        train_dl = DataLoader(train_dataset, train_config["batch_size"], shuffle=True)
        val_dl = DataLoader(val_dataset, train_config["batch_size"], shuffle=True)

        # define evaluation metrics (both for training and validation)
        metrics = (
            {
                "mape": torchmetrics.MeanAbsolutePercentageError().to(device),
            }
            if data_config["regress"]
            else {
                "f1-macro": torchmetrics.F1Score(
                    task="multiclass",
                    average="macro",
                    num_classes=data_config["num_classes"],
                ).to(device),
                "f1-micro": torchmetrics.F1Score(
                    task="multiclass",
                    average="micro",
                    num_classes=data_config["num_classes"],
                ).to(device),
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=data_config["num_classes"],
                ).to(device),
            }
        )

        # instantiate early stopping class
        # if we're doing regression we'll track validation loss
        # if we're doing classification we'll track validation macro-F1
        # this will monitor that metric and save the best-performing model
        early_stopper = EarlyStopping(
            objective="minimize" if data_config["regress"] else "maximize",
            patience=10,
            min_delta=0.01,
            save_path=model_save_dir
            / f"{net.__class__.__name__}_split_{i+1}of{len(cv_splits)}.pt",
        )

        net.to(device)
        for epoch_num, epoch in enumerate(range(train_config["num_epochs"])):
            net, train_metrics = run_dataloader(
                net,
                train_dl,
                epoch,
                train_config["num_epochs"],
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
                train_config["num_epochs"],
                optimizer,
                loss_function,
                metrics,
                train=False,
                device=device,
            )

            early_stopper(
                val_metric=(
                    val_metrics["mape"]
                    if data_config["regress"]
                    else val_metrics["f1-macro"]
                ),
                model=net,
            )

            if early_stopper.early_stop or epoch + 1 == train_config["num_epochs"]:
                results_per_split[i + 1] = {
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "stopped_early": early_stopper.early_stop,
                    "stop_epoch": epoch_num if early_stopper.early_stop else None,
                }
                break

    print(results_per_split)
    with open(f"{model_save_dir}/results_per_split.json", "w") as f:
        json.dump(results_per_split, f, indent=4)

    for fold, data in results_per_split.items():
        if fold not in ["train_config", "data_config"]:
            print(f"Fold {fold}:")
            train_metrics = data["train_metrics"]
            val_metrics = data["val_metrics"]

            print("  Train:", end=" ")
            print(" | ".join(f"{k}: {v:.3f}" for k, v in train_metrics.items()))

            print("  Val:  ", end=" ")
            print(" | ".join(f"{k}: {v:.3f}" for k, v in val_metrics.items()))

            print()


if __name__ == "__main__":

    # declare datasets variables
    data_config = {
        "downsample_img_by": 8,
        "downsample_sequence": True,
        "imgs_per_sec": 3,
        "image_sequences": False,
        "imgs_per_sequence": 5,
        "time_spacing": 1.0,
        "regress": False,
        "dataset_root": (
            # Path(__file__).parents[4] / "srv/evenflow-data/DFKI/Dataset_4_100_traj"
            "/vol/bitbucket/svadakku/data/dfki/Dataset_4_100_traj"
        ),
        "num_classes": 4,  # 2 for regression, 4 for classification
    }

    # get train/val/test splits
    test_videos, cv_splits = cross_validation(
        video_indices=list(range(100)),
        num_test_vids=10,
        num_folds=5,
        seed=42,
    )

    # create CNN
    # net = PretrainedLinear(
    #     pretrained=False, 
    #     num_classes=2 if data_config["regress"] else 10,
    #     softmax=not data_config["regress"],
    # )
    net = RobotNet(
        num_classes=data_config["num_classes"],
        # softmax=not data_config["regress"],
        softmax=False, # we will use argmax to get the class
    )
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")

    cls = net.__class__
    sig = inspect.signature(cls.__init__)
    
    arg_names = list(sig.parameters.keys())[1:]  # skip 'self'
    init_args = {}
    for name in arg_names:
        if hasattr(net, name):
            init_args[name] = getattr(net, name)
        else:
            init_args[name] = None  # or skip it, depending on your preference

    model_info = {
        'model_name': cls.__name__,
        'args': init_args
    }

    # define training config
    train_config = {
        "num_epochs": 100,
        "batch_size": 500,
        "learning_rate": 1e-3,
        "model_info": model_info,
    }

    train(net, data_config, train_config, cv_splits)
