import pytest
import torch
import os 
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA,"processed/cats.pt")), reason="Data files not found")
def test_data():
    dataset = torch.load(os.path.join(_PATH_DATA,"processed/cats.pt"))
    assert len(dataset) == 1706, "Dataset did not have the correct number of samples"
    assert dataset[0].shape == (128,128,3), "Image in the dataset does not have the correct shape"
    assert torch.sum((dataset < 0) | (dataset > 1)).item() == 0, "Image pixels must be in [0;1]"