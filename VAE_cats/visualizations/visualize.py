import torch 
from matplotlib import pyplot as plt 
import numpy as np 
from vae_cats.models.model import Model

# # # # Declare constants # # # #
DATA_PATH = "data/processed/cats.pt"
MODEL_PATH = "models/epoch=4-step=135.ckpt"
NR_OF_CATS = 2000
IMAGE_SIZE = (128, 128, 3)

# init data 
dataset = torch.load(DATA_PATH).float()[:NR_OF_CATS].permute(0, 3, 1, 2)

# init model 
vae = Model.load_from_checkpoint(MODEL_PATH)
vae.eval()

# inference 
with torch.no_grad():
    outputs = vae(dataset).cpu().numpy()

plt.figure(figsize = (10, 8))
plt.hist(outputs.ravel(), bins = 255, color = [247/255, 187/255, 177/255])
plt.savefig("reports/figures/cats_distribution.png")
plt.show()



