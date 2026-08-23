# tests/test_models.py
import pytest
import torch

from hydra_torch.adapters import LinearAdapter
from hydra_torch.backbones import ResNet18, ResNet50
from hydra_torch.heads import IdentityHead
from hydra_torch.models import CIFAR10Model, SimpleModel


def test_cifar10_model_forward_shape():
    model = CIFAR10Model(
        backbone=ResNet50(pretrained=False),
        adapter=LinearAdapter(in_features=2048, out_features=10),
        head=IdentityHead(),
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_simple_model_forward_shape_3_channel():
    """SimpleModel/ResNet18 works fine with 3-channel input — this passes."""
    model = SimpleModel(
        backbone=ResNet18(pretrained=False),
        adapter=LinearAdapter(in_features=512, out_features=10),
        head=IdentityHead(),
    )
    x = torch.randn(2, 3, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


@pytest.mark.xfail(
    reason=(
        "Known unresolved bug: real MNIST images are 1-channel, but "
        "SimpleModel's ResNet18 backbone expects 3. See PLANNER.md "
        "section 1a/4. Remove this xfail once fixed — if it starts "
        "passing unexpectedly, that's your signal the fix landed."
    ),
    strict=True,
)
def test_simple_model_forward_shape_1_channel_real_mnist():
    model = SimpleModel(
        backbone=ResNet18(pretrained=False),
        adapter=LinearAdapter(in_features=512, out_features=10),
        head=IdentityHead(),
    )
    x = torch.randn(2, 1, 28, 28)  # actual MNIST shape
    out = model(x)
    assert out.shape == (2, 10)
