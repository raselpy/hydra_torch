from torchvision import models
from torch import nn


class ResNet18(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained
        print("geting backbone")
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = models.resnet18(weights=weights)
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 10)

    def forward(self, x):
        return self.resnet18(x)