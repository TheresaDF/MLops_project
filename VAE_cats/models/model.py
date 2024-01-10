import torch
from torch import nn 
from pytorch_lightning import LightningModule 


class Model(LightningModule):
    """VAE Model."""

    def __init__(self):
        super(Model, self).__init__()

        self.encode = nn.Sequential(nn.Conv3d(in_channels = 3, out_channels = 4, kernel_size = 3, stride = 1, padding = (1, 1, 0)),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size = 4, stride = 4),
                                    nn.Conv3d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 1, padding = (1, 1, 0)),
                                    nn.ReLU(), 
                                    nn.MaxPool3d(kernel_size = 4, stride = 4))

        self.mean = nn.Linear(in_features = 512, out_features = 32)
        self.var = nn.Linear(in_features = 512, out_features = 32)

        self.decode = nn.Sequential(nn.ConvTranspose3d(in_channels = 8, out_channels = 4, kernel_size= 1, stride = 3, padding = (16, 16)),
                                    nn.ConvTranspose3d(in_channels = 4, out_channels = 3, kernel_size= 1, stride = 3, padding = (16, 16)))

        # self.criterium()

    def forward(self, x):
        """Forward pass."""
        print(f"size of x = {x.shape}")
        x = self.encode(x)
        x = torch.flatten(x)

        print(f"size of x = {x.shape}")

        mu = self.mean(x)
        var = self.mean(x)        

        x = x.view(2, 2, 8)
        print(f"size of x = {x.shape}")

        x = self.decode(x)

        print(f"size of x = {x.shape}")
        return x 

        
