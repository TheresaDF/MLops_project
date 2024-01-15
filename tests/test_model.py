import torch
from vae_cats.models.model import Model

def test_model():
    model = Model(h_dim=1024, z_dim=32)
    output = model(torch.randn((64,3,128,128)))
    
    # test model output shapes
    assert len(output) == 3, "Model expects three outputs: generated image, mean, and variance"
    assert output[0].shape == (64,3,128,128), "Model did not generate correct image shape"
    assert output[1].shape == (64,32), "Model did not have the expected latent size for mean"
    assert output[2].shape == (64,32), "Model did not have the expected latent size for log var"
    assert model.decode(torch.randn((64,1024,1,1))).shape == (64,3,128,128), "Model did not decode noise to correct size"