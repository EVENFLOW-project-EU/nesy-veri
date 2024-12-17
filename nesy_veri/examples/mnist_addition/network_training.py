import os
import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from nesy_veri.neural_utils import run_dataloader


class MNIST_Net(nn.Module):
    def __init__(self, softmax=True, dense_input_size=16 * 4 * 4):
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
            nn.Softmax(dim=-1) if softmax else nn.Identity(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


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
    loss_function = nn.NLLLoss()

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
            device="cpu",
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
            device="cpu",
        )

    # save model parameters to avoid retraining
    torch.save(mnist_net.state_dict(), save_model_path)


def get_mnist_network(model_path: os.PathLike, num_epochs: int = 10):
    mnist_net = MNIST_Net()

    # if the trained network hasn't been saved, train and save it
    if not os.path.exists(model_path):
        train_mnist_network(model_path, num_epochs)

    mnist_net.load_state_dict(torch.load(model_path, weights_only=True))
    mnist_net.eval()

    return mnist_net
