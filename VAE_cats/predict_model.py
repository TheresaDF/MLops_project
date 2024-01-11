import torch 
from pytorch_lightning import Trainer
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.utils import save_image

model = Model.load_from_checkpoint("models/xxx.ckpt")
image = Trainer().test(model)
save_image(image,"test.png")