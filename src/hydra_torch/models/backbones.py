from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet18 = models.resnet18(weights=weights)
        self.resnet18.fc = nn.Identity()

    def forward(self, x):
        return self.resnet18(x)


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet50 = models.resnet50(weights=weights)
        self.resnet50.fc = nn.Identity()

    def forward(self, x):
        return self.resnet50(x)
