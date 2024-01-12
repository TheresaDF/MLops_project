import torch 
from matplotlib import pyplot as plt 
import numpy as np 
from vae_cats.models.model import model 

# # # # Declare constants # # # #
DATA_PATH = "data/processed/cats.pt"
MODEL_PATH = "models/epoch=0-step=27.ckpt"
NR_OF_CATS = 100 
IMAGE_SIZE = (128, 128, 3)

# init data 
images = torch.load(DATA_PATH)
dataset = torch.load("data/processed/cats.pt").float()[:NR_OF_CATS]

# init model 
vae = model()
vae.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])
vae.eval()

# inference 
with torch.no_grad():
    outputs = model(dataset).cpu().numpy()

plt.figure(figsize = (8, 6))
plt.hist(outputs.ravel(), bins = 255, color = [247/255, 187/255, 177/255])
plt.savefig("reports/figures/cats_distribution.png")
plt.show()

