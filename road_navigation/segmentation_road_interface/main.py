from models import UNET , transform
from fastapi import FastAPI , Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi.templating import Jinja2Templates
from models import ImageData
import matplotlib.pyplot as plt
templates = Jinja2Templates(directory="templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/roadSeg")
async def image_get(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/roadSeg")
async def image_recieve(request: Request,imageData : ImageData):
    image = imageData.data
    height = imageData.height
    width = imageData.width
    imageNew = np.array(image).reshape(height,width,4).astype(np.uint8)
    imageNew = imageNew[:,:,:3]
    print(width)
    print(type(image))
    print(imageNew.shape)
    # print(imageNew)
    file_path = "output_image.png"
    plt.imsave(file_path, imageNew)  # Save the image

    # return {"image":image,"height":height,"width":width}
    
