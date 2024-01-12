import pytest
import torch
import os 
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA,"processed/cats.pt")), reason="Data files not found")
def test_data():
    dataset = torch.load(os.path.join(_PATH_DATA,"processed/cats.pt"))
    # check dataset length 
    assert len(dataset) == 1706, "Dataset did not have the correct number of samples"
    # check image shape 
    for img in dataset:
        assert img.shape == (3,128,128), "Image in the dataset does not have the correct shape"