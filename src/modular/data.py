import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class Data:
    def __init__(self,data):
     self.data = data

    def getData(self):
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )


        self.train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

        return self.train_dataloader,self.test_dataloader

    def imgShow(self):
        # Display image and label.
        train_features, train_labels = next(iter(self.train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")