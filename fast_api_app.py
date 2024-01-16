import torch 
from fastapi import FastAPI
from vae_cats.models.model import Model
from fastapi.responses import Response
import io
import imageio
import torchvision

# model = Model.load_from_checkpoint("models\epoch=42-step=1161.ckpt")
model = Model.load_from_checkpoint("models/epoch=0-step=27.ckpt")

app = FastAPI()
@app.get("/")
def show_image():
    with torch.no_grad():
        noise = torch.randn(64, 1024, 1, 1) 
        images = model.decode(noise)
    grid_img = torchvision.utils.make_grid(images, nrow=8).permute(1,2,0).numpy()
    with io.BytesIO() as output:
        imageio.imwrite(output, grid_img, format='PNG')
        data = output.getvalue()
    #bigger size 
    return Response(data, media_type="image/png")


