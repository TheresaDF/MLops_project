import torch 
from pytorch_lightning import Trainer
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import save_image

model = Model.load_from_checkpoint("models/epoch=0-step=27.ckpt")
with torch.no_grad():
    noise = torch.randn(64, 8, 2, 2)
    images = model.decode(noise)
save_image(images.view(64, 3, 128, 128), "generated_sample.png")