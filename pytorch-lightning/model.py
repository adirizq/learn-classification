import torch
import torch.nn as nn
import pytorch_lightning as pl


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