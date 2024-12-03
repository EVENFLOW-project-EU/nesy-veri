import os
import torch
import torchmetrics
from torch import nn, optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18

from nesy_veri.examples.BDD_OIA.bdd_oia_utils import BDDDataset
from nesy_veri.neural_utils import run_dataloader


def train_bdd_network(
    net: nn.Module,
    labels: str,
    save_model_path: os.PathLike,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
):

    train_dataset = BDDDataset(subset="train", labels=labels)
    val_dataset = BDDDataset(subset="val", labels=labels)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    pos_weights = [round(1 / x, 1) for x in train_dataset.get_class_support()]
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weights))
    # loss_function = nn.BCELoss()

    metrics = {
        "f1-macro": torchmetrics.F1Score(
            task="multilabel",
            average="macro",
            num_labels=len(train_dataset[0][1]),
        ),
        "f1-micro": torchmetrics.F1Score(
            task="multilabel",
            average="micro",
            num_labels=len(train_dataset[0][1]),
        ),
        "accuracy": torchmetrics.Accuracy(
            task="multilabel",
            num_labels=len(train_dataset[0][1]),
        ),
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
        )

    # save model parameters to avoid retraining
    torch.save(net.state_dict(), save_model_path)


def load_pretrained_resnet18(num_classes: int):
    # initialize the smallest ResNet from torchvision
    resnet = resnet18()

    # load the weights of this ResNet pretrained for scene tagging
    # in the BDD-100K dataset, this is the big version of our dataset
    # although it is a different task, it's not object detection
    ckpt_path = Path(__file__).parent / "checkpoints/model_checkpoints"
    bdd_100k_weights = torch.load(ckpt_path / "resnet18_4x_scene_tag_bdd100k.pth")[
        "state_dict"
    ]
    state_dict = {
        k.split(".", 1)[1]: bdd_100k_weights[k] for k in bdd_100k_weights.keys()
    }

    # change the final layer of the ResNet so that you can load the weights
    resnet.fc = nn.Linear(in_features=512, out_features=7, bias=True)

    # load the weights
    resnet.load_state_dict(state_dict)

    # change the final layer of the ResNet for our number of classes
    resnet.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    # print("#params:", sum(p.numel() for p in resnet.parameters() if p.requires_grad))

    return resnet


if __name__ == "__main__":
    # load a small ResNet pretrained on scene tagging in the BDD-100K dataset
    resnet = load_pretrained_resnet18(num_classes=21)

    # finetune the network on our task and save the resulting network
    resnet_concepts_path = (
        Path(__file__).parent
        / "checkpoints/model_checkpoints/resnet_concepts_model.pth"
    )
    if not os.path.exists(resnet_concepts_path):
        train_bdd_network(
            net=resnet,
            labels="concepts",
            save_model_path=resnet_concepts_path,
            num_epochs=10,
        )

    # loss: 1.3544 f1-macro: 0.10 f1-micro: 0.17 accuracy: 0.60 @ ~35sec
