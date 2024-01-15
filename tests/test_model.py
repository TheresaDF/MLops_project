import torch
from vae_cats.models.model import Model

def test_model():
    model = Model()
    noise = torch.randn((64,3,128,128))
    # test model output shapes 
    assert len(model(noise)) == 3, "Model expects 3 outputs: generated image, mean, and variance"
    assert model(noise)[0].shape == (64,3,128,128), "Model did not generate correct image shape"
    assert model(noise)[1].shape == (64,32), "Model did not have the expected latent size for mean"
    assert model(noise)[2].shape == (64,32), "Model did not have the expected latent size for log var"
    assert model.decode(torch.randn((64, 1024, 1, 1))).shape == (64,3,128,128), "Model did not decode to correct size"