import torch
import torchvision
import multiprocessing
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=100):
        super().__init__()
        self.batch_size = batch_size


    def load_data(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

        # Split the train dataset into train and validation datasets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        return train_dataset, val_dataset, test_dataset


    def setup(self, stage=None):
        train_data, valid_data, test_data = self.load_data()

        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )


    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )