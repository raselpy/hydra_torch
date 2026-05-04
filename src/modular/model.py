import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        print("getting model")
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone

    def get_model(self):
        model = self.backbone
        return model.to(device)