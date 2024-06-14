import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn

class MLP(L.LightningModule):
    def __init__(
            self,
            dims: list[int],
            mse_alpha,
            corr_alpha):
        super(MLP, self).__init__()

        self.dims = dims

        # First layers
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], dims[i+1]),
                                                   nn.LayerNorm(dims[i+1]),
                                                   nn.ReLU())
                                     for i in range(len(dims) - 2)])

        # Last layer
        self.layers.append(nn.Linear(dims[-2], dims[-1]))

        self.mse_alpha = mse_alpha
        self.corr_alpha = corr_alpha

    def forward(
            self,
            x: torch.tensor):
        for layer in self.layers:
            x = layer(x)
            
        return x

    def custom_loss(
            self,
            y,
            y_hat):
        # MSE
        loss = self.mse_alpha * F.mse_loss(y_hat, y)

        # Normalized cosine: Pearson correlation
        cosine_loss = nn.CosineEmbeddingLoss()
        target = torch.ones(y.shape[1])
        loss += self.corr_alpha * cosine_loss((y_hat - y_hat.mean(dim=0, keepdim=True)).t(),
                                              (y - y.mean(dim=0, keepdim=True)).t(),
                                              target)

        return loss

    def training_step(
            self,
            batch,
            batch_idx):
        x, y = batch

        # Predict
        y_hat = self.forward(x)
        
        loss = self.custom_loss(y, y_hat)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
            self,
            batch,
            batch_idx):
        x, y = batch

        # Predict
        y_hat = self.forward(x)

        val_loss = self.custom_loss(y, y_hat)
        
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(
            self,
            batch,
            batch_idx):
        x, y = batch

        # Predict
        y_hat = self.forward(x)

        test_loss = self.custom_loss(y, y_hat)
        
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(
            self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-4)
        return optimizer
