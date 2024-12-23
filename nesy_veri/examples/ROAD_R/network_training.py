import os
import torch
import torchmetrics
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torchvision.models.resnet import resnet18
from torchvision.models import DenseNet, ResNet18_Weights
from torch.utils.data import DataLoader, random_split

from nesy_veri.examples.ROAD_R import road_utils
from nesy_veri.neural_utils import run_dataloader
from nesy_veri.examples.ROAD_R.road_utils import ROADRPropositional


class ROAD_R_Net(nn.Module):
    def __init__(self, num_classes: int):
        super(ROAD_R_Net, self).__init__()

        # self.size = 8 * 27 * 37
        # self.size = 8 * 12 * 17
        # self.size = 8 * 12 * 17
        # self.size = 16 * 9 * 13
        self.size = 8 * 10 * 14
        # self.size = 8 * 91 * 123
        # self.size = 8 * 21 * 29
        # self.size = 8 * 46 * 62
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            # nn.Conv2d(8, 8, 3),
            # nn.MaxPool2d(2, 2),
            # nn.ReLU(True),
        )

        # self.conv_block1 = nn.Sequential(
        #     nn.Conv2d(3, 16, 10, padding="same"),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(True),
        #     nn.Conv2d(16, 32, 5, padding="same"),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(True),
        # )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(32, 32, 3, padding="same"),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(),
        # )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(32, 16, 3),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 8, 3),
        #     # nn.MaxPool2d(2, 2),
        #     nn.ReLU(),
        # )

        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.size, 50),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(50, num_classes),
            # nn.Linear(self.size, num_classes)
        )

    # def p_forward(self, x):
    #     conv1out = self.conv_block1(x)
    #     from torchvision.transforms import Resize
    #     down = Resize((60, 80))
    #     down2 = Resize((30, 40))

    #     residual = conv1out
    #     conv2out = self.conv_block2(conv1out)
    #     out = conv2out + down(residual)

    #     residual = out
    #     conv2out = self.conv_block2(out)
    #     out = conv2out + down2(residual)

    #     conv3out = self.conv_block3(out)

    #     dense_input = conv3out.view(-1, self.size)
    #     out = self.classifier(dense_input)

    #     return out

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


def train_road_network(
    net: nn.Module,
    labels: str,
    save_model_path: os.PathLike,
    device,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    sample_every_n: int = 50,
    downsample_img_by: int = 2,
):

    net.to(device)

    train_dataset, val_dataset = [
        ROADRPropositional(
            dataset_path=Path(__file__).parents[-2] / "nmanginas/road/dataset",
            subset=subset,
            label_level=labels,
            sample_every_n=sample_every_n,
            downsample_img_by=downsample_img_by,
            balance_feature_dataset=balance,
        )
        for subset, balance in [("train", False), ("val", False)]
    ]
    # dataset = ROADRPropositional(
    #     dataset_path=Path(__file__).parents[-2] / "nmanginas/road/dataset",
    #     subset="train",
    #     label_level=labels,
    #     sample_every_n=sample_every_n,
    #     downsample_img_by=downsample_img_by,
    #     balance_feature_dataset=True,
    # )
    # train_dataset, val_dataset = random_split(dataset, [0.80, 0.20])

    print(f"len(train_dataset): {len(train_dataset)}, len(val_dataset): {len(val_dataset)}")
    print("#params:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    pos_weights = [round(1 / x.item(), 1) for x in train_dataset.get_object_support()]
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weights)).to(
        device
    )
    # loss_function = nn.BCELoss()

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


def get_road_networks(
    object_model_path: os.PathLike,
    action_model_path: os.PathLike,
    num_epochs_objects: int,
    num_epochs_actions: int,
    sample_every_n: int,
    downsample_img_by: int,
    device: torch.device,
):
    # check if the object detection network has already been trained
    # if no, train and save it, else just load it
    object_net = ROAD_R_Net(num_classes=2)
    if not os.path.exists(object_model_path):
        train_road_network(
            net=object_net,
            labels="objects",
            save_model_path=object_model_path,
            device=device,
            num_epochs=num_epochs_objects,
            sample_every_n=sample_every_n,
            downsample_img_by=downsample_img_by,
        )

    object_net.load_state_dict(torch.load(object_model_path, weights_only=True))
    object_net.eval()

    # check if the action selection network has already been trained
    # if no, train and save it, else just load it
    action_net = ROAD_R_Net(num_classes=2)
    if not os.path.exists(action_model_path):
        train_road_network(
            net=action_net,
            labels="actions",
            save_model_path=action_model_path,
            device=device,
            num_epochs=num_epochs_actions,
            sample_every_n=sample_every_n,
            downsample_img_by=downsample_img_by,
        )

    action_net.load_state_dict(torch.load(action_model_path, weights_only=True))
    action_net.eval()

    return object_net, action_net


def get_road_resnets(sample_every_n: int):
    # check if the object detection network has already been trained
    # if no, train and save it, else just load it
    model_dir = Path(__file__).parent / "checkpoints/model_checkpoints"

    object_model_path = model_dir / "resnet_two_objects_div_50_10_epochs.pth"
    if not os.path.exists(object_model_path):
        # object_net = resnet18()
        object_net = resnet18(ResNet18_Weights.DEFAULT)
        object_net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

        train_road_network(
            net=object_net,
            labels="objects",
            save_model_path=object_model_path,
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
            num_epochs=10,
            sample_every_n=sample_every_n,
        )
    else:
        object_net = resnet18()
        object_net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        object_net.load_state_dict(torch.load(object_model_path, weights_only=True))
        object_net.eval()

    # check if the action selection network has already been trained
    # if no, train and save it, else just load it
    action_model_path = model_dir / "resnet_two_actions_div_50_5_epochs.pth"
    if not os.path.exists(action_model_path):
        action_net = resnet18(ResNet18_Weights.DEFAULT)
        action_net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

        train_road_network(
            net=action_net,
            labels="actions",
            save_model_path=action_model_path,
            device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
            num_epochs=5,
            sample_every_n=sample_every_n,
        )
    else:
        action_net = resnet18()
        action_net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        action_net.load_state_dict(torch.load(action_model_path, weights_only=True))
        action_net.eval()

    return object_net, action_net


if __name__ == "__main__":

    torch.manual_seed(0)

    sample_every_n = 12
    downsample_img_by = 5
    num_epochs_objects = 10
    num_epochs_actions = 10
    model_dir = Path(__file__).parent / "checkpoints/model_checkpoints"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    object_model_path = (
        model_dir
        / f"two_objects_div_{sample_every_n}_{downsample_img_by}_{num_epochs_objects}_epochs.pth"
    )
    action_model_path = (
        model_dir
        / f"two_actions_div_{sample_every_n}_{downsample_img_by}_{num_epochs_actions}_epochs.pth"
    )

    object_net, action_net = get_road_networks(
        object_model_path,
        action_model_path,
        num_epochs_objects,
        num_epochs_actions,
        sample_every_n,
        downsample_img_by,
        device,
    )

    # dataset = ROADRPropositional(
    #     dataset_path=Path(__file__).parents[-2] / "nmanginas/road/dataset",
    #     train=True,
    #     label_level="both",
    #     sample_every_n=sample_every_n,
    #     downsample_img_by=2,
    # )

    # a = get_road_resnets(sample_every_n=20)
