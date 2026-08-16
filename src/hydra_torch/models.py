from torch import Tensor, nn


class CIFAR10Model(nn.Module):
    def __init__(self, backbone: nn.Module, adapter: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.adapter(x)
        x = self.head(x)
        return x
