# models/cnn.py

import torch
import torch.nn as nn


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_name: str):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation_name)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class EmotionCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        activation: str = "relu",
        depth: str = "deep",
    ):
        """
        num_classes: عدد الكلاسات (مثلاً 7 لمشاعر FER2013)
        activation: 'relu' أو 'sigmoid' أو 'leakyrelu' أو 'gelu'
        depth: 'shallow' أو 'deep'
        """
        super().__init__()

        in_channels = 1  # صور grayscale
        depth = depth.lower()
        if depth not in ["shallow", "deep"]:
            raise ValueError("depth must be 'shallow' or 'deep'")

        self.depth = depth

        if depth == "shallow":
            # 48x48 → 24x24 → 12x12
            self.features = nn.Sequential(
                ConvBlock(in_channels, 32, activation),
                ConvBlock(32, 64, activation),
            )
            flattened_size = 64 * 12 * 12
            hidden_dim = 256
        else:
            # 48x48 → 24x24 → 12x12 → 6x6 → 3x3
            self.features = nn.Sequential(
                ConvBlock(in_channels, 32, activation),
                ConvBlock(32, 64, activation),
                ConvBlock(64, 128, activation),
                ConvBlock(128, 256, activation),
            )
            flattened_size = 256 * 3 * 3
            hidden_dim = 512

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim),
            get_activation(activation),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
