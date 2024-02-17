import torch
import torchvision
import torch.nn as nn
import multiprocessing
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset
from pytorch_lightning import Trainer


torch.set_float32_matmul_precision('high')


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


class NeuralNet(pl.LightningModule):
    # Initialization
    def __init__(self, input_size, hidden_size, num_classes, lr):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = lr

        self.y_test = torch.tensor([])
        self.y_predicted_test = torch.tensor([])

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    # Forward pass
    def forward(self, x):
        out = self.l1(x.reshape(-1, 28*28))
        out = self.relu(out)
        out = self.l2(out)
        return out


    # Train loop per batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log_dict({'train_loss': loss.item()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    # Validation loop per batch
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log_dict({'val_loss': loss.item()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    # Test batch
    def test_step(self, batch):
        x, y = batch
        outputs = self(x)
        _, predicted = torch.max(outputs, 1)
        self.y_test = torch.cat((self.y_test.to(self.device), y.to(self.device)))
        self.y_predicted_test = torch.cat((self.y_predicted_test.to(self.device), predicted.to(self.device)))


    # Test loop
    def on_test_epoch_end(self):
        n_samples = self.y_test.size(0)
        n_correct = (self.y_predicted_test == self.y_test).sum().item()
        acc = 100.0 * n_correct / n_samples
        self.log_dict({'test_acc': acc}, on_epoch=True, prog_bar=True)


data_module = DataModule(batch_size=100)
model = NeuralNet(input_size=784, hidden_size=500, num_classes=10, lr=0.001)


trainer = Trainer(
    accelerator='gpu',
    max_epochs=5,
    default_root_dir=f'./checkpoints/pytorch-lightnin/mnist',
)

trainer.fit(model, datamodule=data_module)
trainer.test(datamodule=data_module)
