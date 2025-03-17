import torch
from torch import nn
import torch.nn.functional as F


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


class CNN3D(nn.Module):
    """
    3D CNN for processing (T, C, H, W) inputs and classifying into 13 categories.

    Input shape: (batch_size, 3, 5, 60, 80)
    Output: 13-class probability distribution
    """

    def __init__(self, num_classes=13):
        super(CNN3D, self).__init__()

        # Conv3D: (in_channels=3, out_channels=16, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        # Conv3D: (16 -> 32 filters)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        # Conv3D: (32 -> 64 filters)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        # Adaptive pooling to reduce to fixed size
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)  # Shape: (batch_size, 64, 1, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Shape: (batch_size, 64)

        x = self.classifier(x)  # Shape: (batch_size, num_classes)
        return x 
