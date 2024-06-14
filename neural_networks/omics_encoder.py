import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# Based on: https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html
class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(input_size, int(input_size * 0.75)),
                                nn.LayerNorm(int(input_size * 0.75)),
                                nn.ReLU(),
                                nn.Linear(int(input_size * 0.75), int(input_size * 0.5)),
                                nn.LayerNorm(int(input_size * 0.5)))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(int(output_size * 0.5), int(output_size * 0.75)),
                                nn.LayerNorm(int(output_size * 0.75)),
                                nn.ReLU(),
                                nn.Linear(int(output_size * 0.75), output_size))

    def forward(self, x):
        return self.l1(x)

class AE(L.LightningModule):
    def __init__(self, encoder, decoder, input_size, output_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mlp = nn.Sequential(nn.Linear(int(input_size * 0.5), int(input_size * 0.25)),
                                 nn.LayerNorm(int(input_size * 0.25)),
                                 nn.ReLU(),
                                 nn.Linear(int(input_size * 0.25), output_size))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.mlp(z)

        return x_hat, y_hat

    def custom_loss(self, y, y_hat):
        # MSE
        loss = 0.5 * F.mse_loss(y_hat, y)

        # Normalized cosine = Pearson correlation
        cosine_loss = nn.CosineEmbeddingLoss()
        target = torch.ones(y.shape[1])
        loss += 0.5 * cosine_loss((y_hat - y_hat.mean(dim=0, keepdim=True)).t(),
                                  (y - y.mean(dim=0, keepdim=True)).t(),
                                  target)

        return loss

    def custom_combined_loss(self,
                             y, y_hat,
                             x, x_hat):
        regression_loss = self.custom_loss(y, y_hat)
        reconstruction_loss = self.custom_loss(x, x_hat)

        return regression_loss + reconstruction_loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        
        loss = self.custom_combined_loss(y, y_hat,
                                         x, x_hat)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        
        val_loss = self.custom_combined_loss(y, y_hat,
                                             x, x_hat)
        
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        
        test_loss = self.custom_combined_loss(y, y_hat,
                                              x, x_hat)
        
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-4)
        return optimizer
