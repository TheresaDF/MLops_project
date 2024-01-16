import torch 
from pytorch_lightning import Trainer
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra

@hydra.main(config_path="../conf", config_name="config.yaml",version_base=None)
def train(cfg) -> None:
    """ Training the VAE model using pytorch lightning. """
    # set up 
    hparams = cfg.experiments
    data_params = cfg.data
    dataset = DataLoader(torch.load("data/processed/cats.pt").float(), batch_size=hparams.batch_size, shuffle=True, num_workers = 4)
    
    model = Model(image_channels=data_params.channels, h_dim=hparams.h_dim, z_dim=hparams.z_dim, lr=hparams.lr)
    
    # save best model 
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="train_loss", mode="min"
    )
    # add early stopping
    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=10, verbose=True, mode="min"
    )
    # monitor the learning rate 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # train
    trainer = Trainer(callbacks=[early_stopping_callback, checkpoint_callback,lr_monitor],logger=pl.loggers.WandbLogger(project="dtu_mlops"),log_every_n_steps=10, max_epochs=hparams.max_epochs)
    trainer.fit(model,dataset)
    
if __name__ == "__main__":
    train() 