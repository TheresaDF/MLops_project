import glob 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import torch 
import numpy as np 
from tqdm import tqdm 
from matplotlib import pyplot as plt 

# # # Declare constants # # # 
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed/cats.pt"
OUTPUT_SIZE = (128, 128, 3)

# get all filenames 
raw_files = glob.glob(os.path.join(RAW_PATH, '*jpg'))
N = len(raw_files)
all_images = np.zeros((N, OUTPUT_SIZE[0], OUTPUT_SIZE[1], OUTPUT_SIZE[2]))

# read and resize images 
for i, filename in tqdm(enumerate(raw_files)):
    all_images[i] = resize(imread(filename), output_shape=OUTPUT_SIZE)

# save in correct format 
all_images = torch.from_numpy(all_images)
torch.save(all_images, PROCESSED_PATH)
