"""Model definitions for the CIFAR-10 image classifier.

Provides a small custom CNN and a CIFAR-adapted ResNet-18, selected at
runtime via the `architecture` field in configs/training_config.yaml.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

SUPPORTED_ARCHITECTURES = ("simple_cnn", "resnet18")


class SimpleCNN(nn.Module):
    """A small 4-layer CNN, useful as a fast-training baseline on CPU."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet-18 adapted for 32x32 inputs (CIFAR-10 / Fashion-MNIST scale).

    Standard torchvision ResNet-18 was designed for 224x224 ImageNet images.
    Its 7x7 stride-2 stem + maxpool would shrink a 32x32 image to ~2x2
    before the residual blocks even start, destroying spatial information.
    The common fix (used by most CIFAR ResNet implementations) is to swap
    the stem for a 3x3 stride-1 conv and drop the initial maxpool.
    """
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(architecture: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Factory used by train.py and serve.py so both stay in sync."""
    architecture = architecture.lower()
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if architecture == "resnet18":
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unknown architecture '{architecture}'. Supported: {SUPPORTED_ARCHITECTURES}"
    )
