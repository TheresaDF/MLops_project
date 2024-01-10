import torch
from torch import nn 
from pytorch_lightning import LightningModule 


class Model(LightningModule):
    """VAE Model."""

    def __init__(self):
        super(Model, self).__init__()

        self.encode = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 4, kernel_size = 3, stride = 1, padding = (1, 1)),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 4, stride = 4),
                                    nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = (1, 1)),
                                    nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size = 4, stride = 4))

        self.FC_mean = nn.Linear(in_features = 512, out_features = 32)
        self.FC_var = nn.Linear(in_features = 512, out_features = 32)

        self.decode = nn.Sequential(nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size= 3, stride = 16, padding = (0, 0), output_padding = (13, 13)),
                                    nn.ConvTranspose2d(in_channels = 4, out_channels = 3, kernel_size= 3, stride = 4, padding = (0, 0), output_padding = (1, 1)))

        self.mean = -1
        self.log_var = -1 

    def forward(self, x):
        """Forward pass."""
        h_ = self.encode(x)
        h_ = h_.view(-1, 512)

        print(f"shape of h_ = {h_.shape}")

        self.mean = self.FC_mean(h_)
        self.log_var = self.FC_var(h_)        
        std = torch.exp(0.5 * self.log_var)
        z = self.reparam(self.mean, std)

        z = z.view(-1, 8, 2, 2)
        print(f"shape of z = {z.shape}")
        x_hat = self.decode(z)

        print(f"shaoe of hat {x_hat.shape}")

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

if __name__ == "__main__":
    model = Model()
    model(torch.randn((64, 3, 128, 128)))

        
