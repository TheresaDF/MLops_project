import torch
from torch import nn, optim 
from pytorch_lightning import LightningModule 
import wandb
import matplotlib.pyplot as plt

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.reshape(input.size(0), size, 1, 1)


class Model(LightningModule):
    """VAE Model."""

    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(Model, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=8),
            nn.ReLU(),
            Flatten())
        
        self.FC_mean = nn.Linear(h_dim, z_dim)
        self.FC_var = nn.Linear(h_dim, z_dim)
        self.fc = nn.Linear(z_dim, h_dim)
        
        self.decode = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=2, stride=2),
            nn.Sigmoid())
        self.criterium = self.loss_function
    
    def reparam(self, h):
        mu, logvar = self.FC_mean(h), self.FC_var(h)
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar

    def forward(self, x):
        """Forward pass."""
        h_ = self.encode(x)
        z,mu,logvar = self.reparam(h_)
        z = self.fc(z)
        return self.decode(z), mu, logvar
    
    def loss_function(self, x, x_hat, mean, log_var):
        """Elbo loss function."""
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.permute(0, 3, 1, 2)
        x_hat,mean,logvar = self(x)
        if batch_idx % 100 == 0:
            self.logger.experiment.log({"reconstruction": [wandb.Image(x_hat[0], caption="reconstruction")]})
            self.logger.experiment.log({"real": [wandb.Image(x[0], caption="real")]})
        loss = self.criterium(x, x_hat,mean,logvar)
        self.log("train_loss", loss)


        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss", "interval": 1}