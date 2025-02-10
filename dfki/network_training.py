import os
import torch
import torchmetrics
from pathlib import Path
from torch import nn, optim
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

from dfki.new_data import DetectedRobotImages


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

    return network


class RobotNet(nn.Module):
    def __init__(self, num_classes: int, softmax: bool):
        super(RobotNet, self).__init__()

        self.size = 8 * 8 * 17
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 10),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.size, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Softmax() if softmax else nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    # declare datasets variables
    downsample_img_by = 8
    downsample_sequence = True
    imgs_per_sec = 2
    dataset_root = Path(__file__).parents[4] / "srv/evenflow-data/DFKI/Dataset_2"

    train_dataset = DetectedRobotImages(downsample_img_by, downsample_sequence, imgs_per_sec, "train", dataset_root)
    val_dataset = DetectedRobotImages(downsample_img_by, downsample_sequence, imgs_per_sec, "val", dataset_root)

    # create CNN for training
    net = RobotNet(num_classes=len(train_dataset[0][1]), softmax=True)

    # define training config
    lr = 1e-3
    batch_size = 32
    num_epochs = 10
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

    net.to(device)
    for epoch in range(num_epochs):
        net = run_dataloader(
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

        net = run_dataloader(
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

    # # save model parameters to avoid retraining
    # torch.save(net.state_dict(), save_model_path)

    # return net
