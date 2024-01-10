import torch
from VAE_cats.models.model import Model

def test_model():
    # # load model 
    model = Model()
    # test model 
    assert model(torch.randn((64,3,128,128))).shape == (64,3,128,128), "Model did not output the expected/correct shape"