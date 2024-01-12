import torch 
from pytorch_lightning import Trainer
from vae_cats.models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

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
        noise = torch.randn(64, 8, 2, 2)
        images = model.decode(noise)
    save_image(images.view(64, 3, 128, 128), "reports/figures/generated_sample.png")
    
if __name__ == "__main__":
    predict("models/epoch=4-step=135.ckpt")