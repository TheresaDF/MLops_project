import torch 
from fastapi import FastAPI
from vae_cats.models.model import Model
from fastapi.responses import Response
from io import BytesIO
from PIL import Image
import numpy as np
import torchvision

model = Model.load_from_checkpoint("models\epoch=42-step=1161.ckpt")
# model = Model.load_from_checkpoint("models/epoch=0-step=27.ckpt")

app = FastAPI()
@app.get("/")
def show_image():
    with torch.no_grad():
        noise = torch.randn(64, 1024, 1, 1) 
        images = model.decode(noise)
    grid_img = torchvision.utils.make_grid(images, nrow=8).permute(1,2,0).numpy()
    pil_img = Image.fromarray((grid_img * 255).astype(np.uint8))

    img_byte_array = BytesIO()
    pil_img.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    return Response(img_byte_array.read(), media_type="image/png")