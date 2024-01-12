import glob 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import torch 
import numpy as np 
from tqdm import tqdm 
import hydra

# RAW_PATH = "data/raw"
# PROCESSED_PATH = "data/processed/cats.pt"
# OUTPUT_SIZE = (128, 128, 3)

@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def process_data(cfg) -> None:
    """ Pre-processes the raw data of cat images. """
    data_params = cfg.data
    output_size = (data_params.xy_dim, data_params.xy_dim, data_params.channels)
  
    # get all filenames 
    raw_files = glob.glob(os.path.join(data_params.data_path_raw, '*jpg'))
    N = len(raw_files)
    all_images = np.zeros((N, output_size[0], output_size[1], output_size[2]))

    # read and resize images 
    for i, filename in tqdm(enumerate(raw_files)):
        all_images[i] = resize(imread(filename), output_shape=output_size)

    # save in correct format  
    all_images = torch.from_numpy(all_images)
    torch.save(all_images, data_params.data_path_processed)

if __name__ == "__main__":
    process_data()

