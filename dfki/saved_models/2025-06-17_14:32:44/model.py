import json
from pathlib import Path
from torch import nn


class RobotNet(nn.Module):
    def __init__(self, num_classes: int, softmax: bool):
        super(RobotNet, self).__init__()

        self.size = 16 * 2 * 6
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 10),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(True),
            nn.Conv2d(256, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.size, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Softmax() if softmax else nn.Identity(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = RobotNet(num_classes=2, softmax=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    f = open(Path(__file__).parent / "results_per_split.json")
    data = json.load(f)

    for key, metrics in data.items():
        if key not in ["train_config", "data_config"]:
            print(key)
            print("MAPE (train):", round(metrics["train_metrics"]["mape"], 2))
            print("MAPE (val):  ", round(metrics["val_metrics"]["mape"], 2))
            print()

    print()
    for key, metrics in data.items():
        if key not in ["train_config", "data_config"]:
            print(round(metrics["val_metrics"]["mape"], 2))
