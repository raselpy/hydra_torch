import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        data_dir: str = "./",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.mnist_train, True)

    def val_dataloader(self):
        return self._loader(self.mnist_val, False)

    def test_dataloader(self):
        return self._loader(self.mnist_test, False)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        data_dir: str = "./",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [0.9, 0.1])
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.cifar10_train, True)

    def val_dataloader(self):
        return self._loader(self.cifar10_val, False)

    def test_dataloader(self):
        return self._loader(self.cifar10_test, False)
