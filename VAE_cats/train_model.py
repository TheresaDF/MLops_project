import torch 
from pytorch_lightning import Trainer
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader

dataset = DataLoader(torch.load("data/processed/cats.pt").float(), batch_size=64, shuffle=True)
model = Model()
checkpoint_callback = ModelCheckpoint(
    dirpath="./models", monitor="train_loss", mode="min"
)
early_stopping_callback = EarlyStopping(
    monitor="train_loss", patience=3, verbose=True, mode="min"
)
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(callbacks=[early_stopping_callback, checkpoint_callback,lr_monitor],logger=pl.loggers.WandbLogger(project="dtu_mlops"),log_every_n_steps=1)
trainer.fit(model,dataset)