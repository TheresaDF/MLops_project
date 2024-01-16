import torch 
from models.model import Model
from torchvision.utils import save_image
import os
import argparse

def predict(model_path: str) -> None:
    """ Generate cat images from white noise using a trained model. 
    
        Args:
            model_path: the directory to a trained model to make predictions with. 
            
        Returns: 
            None
    """
    if not os.path.exists(model_path):
        return
    model = Model.load_from_checkpoint(model_path)
    with torch.no_grad():
        noise = torch.randn(64, 1024, 1, 1)
        images = model.decode(noise)
    save_image(images.view(64, 3, 128, 128), "reports/figures/generated_sample.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "models/epoch=0-step=27.ckpt", type = str)
    args = parser.parse_args()

    predict(args.model)