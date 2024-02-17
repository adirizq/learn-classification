import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from data_loader import DataModule
from model import NeuralNet

torch.set_float32_matmul_precision('high')

data_module = DataModule(batch_size=100)
model = NeuralNet(input_size=784, hidden_size=500, num_classes=10, lr=0.001)

trainer = Trainer(
    accelerator='gpu',
    max_epochs=5,
    default_root_dir=f'./checkpoints/pytorch-lightning/mnist',
)

trainer.fit(model, datamodule=data_module)
trainer.test(datamodule=data_module)
