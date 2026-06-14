import torch
import torch.nn as nn


ARCHITECTURE_VERSION = "fashion-resnet-v1"
FEATURE_VERSION = "prelogits-128-v1"
FEATURE_DIM = 128
INPUT_SHAPE = (1, 32, 32)
INPUT_RANGE = (0.0, 1.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class FashionMNISTClassifier(nn.Module):
    architecture_version = ARCHITECTURE_VERSION
    feature_version = FEATURE_VERSION
    num_features = FEATURE_DIM

    def __init__(self, mean, std, num_classes=10):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.numel() != 1 or std.numel() != 1:
            raise ValueError("FashionMNIST normalization must be scalar")
        mean = mean.reshape(1, 1, 1, 1)
        std = std.reshape(1, 1, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Normalization mean/std must be finite")
        if std.item() <= 0:
            raise ValueError("Normalization std must be positive")

        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, dropout=0.05),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, dropout=0.05),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, dropout=0.10),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 384, stride=2),
            ResidualBlock(384, 384, dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_head = nn.Sequential(
            nn.Linear(384, FEATURE_DIM, bias=False),
            nn.BatchNorm1d(FEATURE_DIM),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    @staticmethod
    def validate_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Classifier input must be a torch.Tensor")
        if x.dtype != torch.float32:
            raise TypeError(
                f"Classifier input must be float32, received {x.dtype}"
            )
        if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(
                "Classifier input must have shape [B, 1, 32, 32], "
                f"received {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Classifier input contains non-finite values")
        min_value, max_value = torch.aminmax(x)
        if min_value.item() < INPUT_RANGE[0] or max_value.item() > INPUT_RANGE[1]:
            raise ValueError(
                "Classifier input must be in [0, 1], "
                f"received [{min_value.item():.6g}, {max_value.item():.6g}]"
            )

    def extract_features(self, x):
        self.validate_input(x)
        x = (x - self.input_mean) / self.input_std
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.feature_head(x)

    def forward(self, x):
        return self.classifier(self.extract_features(x))
