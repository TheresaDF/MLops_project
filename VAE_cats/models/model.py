import torch
from torch import nn, optim 
from pytorch_lightning import LightningModule 



class Model(LightningModule):
    """VAE Model."""

    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = 3, stride = 1, padding = (1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 4, stride = 4),
                                    nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = (1, 1)),
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size = 4, stride = 4))

        self.FC_mean = nn.Linear(in_features = 512, out_features = 32)
        self.FC_var = nn.Linear(in_features = 512, out_features = 32)

        self.decode = nn.Sequential(nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size= 3, stride = 16, padding = (0, 0), output_padding = (13, 13)),
                                    nn.ConvTranspose2d(in_channels = 4, out_channels = 3, kernel_size= 3, stride = 4, padding = (0, 0), output_padding = (1, 1)),
                                    nn.Sigmoid())

        self.mean = -1
        self.log_var = -1 

    def forward(self, x):
        """Forward pass."""
        h_ = self.encode(x)
        h_ = h_.view(-1, 512)

        self.mean = self.FC_mean(h_)
        self.log_var = self.FC_var(h_)        
        std = torch.exp(0.5 * self.log_var)
        z = self.reparam(self.mean, std)

        z = z.view(-1, 8, 2, 2)
        x_hat = self.decode(z)

        return x_hat

    
    def reparam(self, mean, std):
        epsilon = torch.rand_like(std)
        z = mean + std * epsilon
        return z 
    
    def loss_function(self, x, x_hat):
        """Elbo loss function."""
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + self.log_var - self.mean.pow(2) - self.log_var.exp())
        return reproduction_loss + kld
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.view(-1, 3, 128, 128)
        x_hat = self(x)
        loss = self.loss_function(x, x_hat)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        # we have self.parameters?
        return optim.Adam(self.parameters(), lr = 1e-2)