import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b0


class RobotNet(nn.Module):
    def __init__(self, num_classes: int, softmax: bool):
        super(RobotNet, self).__init__()

        self.size = 8 * 3 * 7
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
            # nn.Softmax() if softmax else nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


class PretrainedLinearOld(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedLinearOld, self).__init__()
        self.pretrained = efficientnet_b0(weights="IMAGENET1K_V1")

        self.linear = nn.Sequential(
            nn.Linear(1000, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        proj = self.pretrained(x)
        final = self.linear(proj)
        return final


class PretrainedLinear(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedLinear, self).__init__()

        # Load pretrained EfficientNet-B0 and remove classification head
        base_model = efficientnet_b0(weights="IMAGENET1K_V1")
        self.feature_extractor = base_model.features  # CNN backbone
        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Convert to (B, 1280, 1, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):  # x: (B, 3, H, W)
        features = self.feature_extractor(x)  # -> (B, 1280, H', W')
        avg_pool = self.global_avg_pool(features)  # -> (B, 1280, 1, 1)
        flatten = avg_pool.view(avg_pool.size(0), -1)  # -> (B, 1280)
        outputs = self.classifier(flatten)  # -> (B, num_classes)
        return outputs


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


class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        base_model = efficientnet_b0(weights="IMAGENET1K_V1")
        self.pretrained_cnn = nn.Sequential(
            *list(base_model.children())[:-2]
        )  # Remove classification head

        # Get feature size from EfficientNet output
        dummy_input = torch.randn(1, 3, 90, 160)
        feature_size = self.pretrained_cnn(dummy_input).view(1, -1).size(1)

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )

        self.linear = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        batch_size, _, seq_len, _, _ = x.shape  # (B, 5, 3, 224, 224)

        # Extract features for each image in the sequence
        cnn_features = []
        for t in range(seq_len):
            features = self.pretrained_cnn(
                x[:, :, t, :, :]
            )  # Pass each image through CNN
            features = features.view(batch_size, -1)  # Flatten
            cnn_features.append(features)

        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (B, 5, feature_size)

        # LSTM
        lstm_out, _ = self.lstm(cnn_features)  # Shape: (B, 5, hidden_size)
        last_output = lstm_out[:, -1, :]  # Take the output at the last timestep

        # Classification
        logits = self.linear(last_output)  # Shape: (B, num_classes)
        return logits
