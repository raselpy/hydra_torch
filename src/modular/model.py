import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    # def __init__(self, num_classes: int, backbone: nn.Module):
    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        self.num_classes = num_classes
        # self.backbone = backbone

    def get_model(self):
        print("getting model")
        self.pretrained = True
        weights = None
        if self.pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = models.resnet18(weights=weights)
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 10)
        return resnet18().to(device)