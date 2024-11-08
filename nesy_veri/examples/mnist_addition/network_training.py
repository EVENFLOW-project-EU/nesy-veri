import os
from pathlib import Path
import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, mnist

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class MNIST_Net(nn.Module):
    def __init__(self, dense_input_size=16 * 4 * 4):
        super(MNIST_Net, self).__init__()

        self.size = dense_input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.size, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            # nn.Softmax(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


def run_dataloader(
    network,
    dataloader,
    epoch,
    num_epochs,
    optimizer,
    loss_function,
    metrics,
    train,
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

    with progress_bar as progress:
        for idx, (inputs, labels) in enumerate(dataloader):
            outputs = network(inputs)

            loss = loss_function(outputs, labels)

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


def train_mnist_network(
    save_model_path: os.PathLike,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = MNIST(root="data/", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="data/", train=False, download=True, transform=transform)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

    mnist_net = MNIST_Net()
    optimizer = optim.Adam(mnist_net.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    metrics = {
        "f1-macro": torchmetrics.F1Score(
            task="multiclass",
            average="macro",
            num_classes=10,
        ),
        "f1-micro": torchmetrics.F1Score(
            task="multiclass",
            average="micro",
            num_classes=10,
        ),
        "accuracy": torchmetrics.Accuracy(
            task="multiclass",
            num_classes=10,
        ),
    }

    for epoch in range(num_epochs):
        mnist_net = run_dataloader(
            mnist_net,
            train_dl,
            epoch,
            num_epochs,
            optimizer,
            loss_function,
            metrics,
            train=True,
        )
        mnist_net = run_dataloader(
            mnist_net,
            test_dl,
            epoch,
            num_epochs,
            optimizer,
            loss_function,
            metrics,
            train=False,
        )

    # save model parameters to avoid retraining
    torch.save(mnist_net.state_dict(), save_model_path)


def get_mnist_network(model_path: os.PathLike):
    mnist_net = MNIST_Net()

    # if the trained network hasn't been saved, train and save it
    if not os.path.exists(model_path):
        train_mnist_network(save_model_path=model_path, num_epochs=10)

    mnist_net.load_state_dict(torch.load(model_path, weights_only=True))
    mnist_net.eval()

    return mnist_net


if __name__ == "__main__":

    model_path = Path(__file__).parent.resolve() / "checkpoints" / "trained_model.pth"
    mnist_net = get_mnist_network(model_path=model_path)

    print()
