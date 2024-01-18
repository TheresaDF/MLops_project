import torch
from typing import Union
from torch import nn, Tensor
from pytorch_lightning import LightningModule 
import wandb



class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.reshape(input.size(0), size, 1, 1)


class Model(LightningModule):
    """Initialize a VAE Model using Pytorch Lightning. 
    
        Args: 
            image_channels = int (number og channels in the image)
            h_dim = int (hidden dimension in the VAE)
            z_dim = int (latent dimension in the VAE)
            lr = int (learning rate during training) """

    def __init__(self, image_channels: int=3, h_dim: int=1024, z_dim: int=32, lr: float=1e-2):
        super(Model, self).__init__()
        
        self.lr = lr
        
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
        # self.kornia_loss = kornia.losses.kl_div_loss_2d()
    
    def reparam(self, h: Tensor) -> Union[Tensor, Tensor, Tensor]:
        """ Reparameterization of the hidden variable. """
        mu, logvar = self.FC_mean(h), self.FC_var(h)
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z, mu, logvar

    def forward(self, x: Tensor) -> Union[Tensor, Tensor, Tensor]:
        """Forward pass."""
        h_ = self.encode(x)
        z,mu,logvar = self.reparam(h_)
        z = self.fc(z)
        return self.decode(z), mu, logvar
    
    def loss_function(self, x: Tensor, x_hat: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
        """Elbo loss function (reproduction loss + Kullback-Leibler divergence). """
        reproduction_loss = nn.functional.mse_loss(x, x_hat, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld
    
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """ The training step, relevant for training using Pytorch Lightning."""
        x = batch
        x = x.permute(0, 3, 1, 2)   # dim = (batch_size, channels, x_dim, y_dim)
        # forward pass 
        x_hat,mean,logvar = self(x)
        if batch_idx % 100 == 0:
            # log information to weights and biases
            self.logger.experiment.log({"reconstruction": [wandb.Image(x_hat[0], caption="reconstruction")]})
            self.logger.experiment.log({"real": [wandb.Image(x[0], caption="real")]})
        # calculate loss 
        loss = self.criterium(x, x_hat,mean,logvar)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self) -> dict:
        """ Obtain Adam optimizer and learning rate scheduler. """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss", "interval": 1}