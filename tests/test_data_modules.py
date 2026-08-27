# tests/test_data_modules.py
import torch
from torch.utils.data import Dataset

from hydra_torch.data.data_modules import CIFAR10DataModule, MNISTDataModule


class _FakeImageDataset(Dataset):
    """Lazy fake dataset — no data actually generated until __getitem__ is
    called, so a large `n` (e.g. matching MNIST's real 60000) costs nothing
    at construction time."""

    def __init__(self, n: int, channels: int, size: int):
        self.n = n
        self.channels = channels
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(self.channels, self.size, self.size), 0


def test_mnist_data_module_batch_shape(monkeypatch):
    # MNISTDataModule.setup() hardcodes random_split(..., [55000, 5000]),
    # which requires exactly 60000 total samples — matching that here so
    # the real (unmodified) setup() logic runs against fake data.
    def fake_mnist(root, train=True, download=False, transform=None):
        n = 60000 if train else 10000
        return _FakeImageDataset(n=n, channels=1, size=28)

    monkeypatch.setattr("hydra_torch.data.data_modules.MNIST", fake_mnist)

    dm = MNISTDataModule(batch_size=4, data_dir="/tmp/fake_mnist")
    dm.setup("fit")
    dm.setup("test")

    images, labels = next(iter(dm.train_dataloader()))
    assert images.shape[1:] == (1, 28, 28)
    assert images.shape[0] <= 4
    assert labels.shape[0] == images.shape[0]

    images, labels = next(iter(dm.test_dataloader()))
    assert images.shape[1:] == (1, 28, 28)


def test_cifar10_data_module_batch_shape(monkeypatch):
    def fake_cifar10(root, train=True, download=False, transform=None):
        n = 500 if train else 100  # fractional split, any size works
        return _FakeImageDataset(n=n, channels=3, size=32)

    monkeypatch.setattr("hydra_torch.data.data_modules.CIFAR10", fake_cifar10)

    dm = CIFAR10DataModule(batch_size=4, data_dir="/tmp/fake_cifar10")
    dm.setup("fit")
    dm.setup("test")

    images, labels = next(iter(dm.train_dataloader()))
    assert images.shape[1:] == (3, 32, 32)
    assert images.shape[0] <= 4
    assert labels.shape[0] == images.shape[0]

    images, labels = next(iter(dm.test_dataloader()))
    assert images.shape[1:] == (3, 32, 32)
