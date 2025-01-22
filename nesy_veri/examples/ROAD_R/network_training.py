import os
import torch
import torchmetrics
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from nesy_veri.examples.ROAD_R import road_utils
from nesy_veri.neural_utils import run_dataloader
from nesy_veri.examples.ROAD_R.road_utils import ROADRPropositional


class ROAD_R_Net(nn.Module):
    def __init__(self, num_classes: int, softmax: bool):
        super(ROAD_R_Net, self).__init__()

        self.size = 8 * 12 * 17
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


def train_road_network(
    labels: str,
    sample_every_n: int,
    downsample_img_by: int,
    save_model_path: os.PathLike,
    device: torch.device,
    num_epochs: int,
    batch_size: int = 32,
    lr: float = 1e-3,
):

    net = ROAD_R_Net(
        num_classes=2,
        softmax=False if labels == "objects" else True,
    )

    dataset = ROADRPropositional(
        dataset_path=Path(__file__).parents[3] / "dataset",
        subset="all",
        label_level=labels,
        sample_every_n=sample_every_n,
        downsample_img_by=downsample_img_by,
        balance_feature_dataset=False,
    )

    gen = torch.Generator()
    gen.manual_seed(0)
    train_dataset, _ = random_split(dataset, [0.8, 0.2], generator=gen)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=gen)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_function = nn.BCELoss() if labels == "objects" else nn.NLLLoss()

    metrics = {
        "f1-macro": torchmetrics.F1Score(
            task="multilabel",
            average="macro",
            num_labels=len(train_dataset[0][1]),
        ).to(device),
        "f1-micro": torchmetrics.F1Score(
            task="multilabel",
            average="micro",
            num_labels=len(train_dataset[0][1]),
        ).to(device),
        "accuracy": torchmetrics.Accuracy(
            task="multilabel",
            num_labels=len(train_dataset[0][1]),
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

    # save model parameters to avoid retraining
    torch.save(net.state_dict(), save_model_path)

    return net


def get_road_network(
    model_dir: os.PathLike,
    labels: str,
    sample_every_n: int,
    downsample_img_by: int,
    device: torch.device,
    num_epochs: int,
):

    filename = f"{labels}_{sample_every_n}_{downsample_img_by}_{num_epochs}.pth"
    model_path = model_dir / filename # type: ignore

    if os.path.exists(model_path):
        net = ROAD_R_Net(
            num_classes=2,
            softmax=False if labels == "objects" else True,
        )
        net.load_state_dict(torch.load(model_path, weights_only=True))
        net.eval()
    else:
        net = train_road_network(
            labels,
            sample_every_n,
            downsample_img_by,
            model_path,
            device,
            num_epochs,
        )

    return net


if __name__ == "__main__":

    sample_every_n = 24
    downsample_img_by = 4
    num_epochs_objects = 20
    num_epochs_actions = 10
    model_dir = Path(__file__).parent / "checkpoints/model_checkpoints"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    object_net = get_road_network(
        model_dir,
        "objects",
        sample_every_n,
        downsample_img_by,
        device,
        num_epochs_objects,
    )

    action_net = get_road_network(
        model_dir,
        "actions",
        sample_every_n,
        downsample_img_by,
        device,
        num_epochs_actions,
    )