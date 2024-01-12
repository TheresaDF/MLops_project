import torch
from torch import nn, optim 
from pytorch_lightning import LightningModule 
import wandb
import matplotlib.pyplot as plt



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
            nn.ReLU()
        )
        
        self.FC_mean = nn.Linear(h_dim, z_dim)
        self.FC_var = nn.Linear(h_dim, z_dim)
        self.fc = nn.Linear(z_dim, h_dim)
        
        self.decode = nn.Sequential(
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

    def forward(self, x):
        """Forward pass."""
        h_ = self.encode(x)
        h_ = h_.reshape(h_.size(0),-1)

        self.mean = self.FC_mean(h_)
        self.log_var = self.FC_var(h_)        
        std = torch.exp(0.5 * self.log_var)
        z = self.reparam(self.mean, std)

        z = self.fc(z)
        z = z.reshape(z.size(0), 1024, 1, 1)
        return self.decode(z)

    
    def reparam(self, mean, std):
        epsilon = torch.rand_like(std)
        z = mean + std * epsilon
        return z 
    
    def loss_function(self, x, x_hat):
        """Elbo loss function."""
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + self.log_var - self.mean.pow(2) - self.log_var.exp())
        return reproduction_loss + kld
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.permute(0, 3, 1, 2)
        x_hat = self(x)
        if batch_idx % 100 == 0:
            self.logger.experiment.log({"reconstruction": [wandb.Image(x_hat[0], caption="reconstruction")]})
            self.logger.experiment.log({"real": [wandb.Image(x[0], caption="real")]})
        loss = self.criterium(x, x_hat)
        self.log("train_loss", loss)


        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss", "interval": 1}