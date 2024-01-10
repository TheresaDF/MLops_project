import torch 
from pytorch_lightning import Trainer
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader

dataset = DataLoader(torch.load("data/processed/cats.pt"), batch_size=64, shuffle=True)
model = Model()
checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="val_loss", mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
)
trainer = Trainer(callbacks=[early_stopping_callback, checkpoint_callback],logger=pl.loggers.WandbLogger(project="dtu_mlops"))
trainer.fit(model,dataset)
trainer.save_checkpoint("models/model.ckpt")