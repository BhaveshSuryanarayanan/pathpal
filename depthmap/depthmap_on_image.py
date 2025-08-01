import numpy as np
import time
import cv2 
import torch 
from enum import Enum
import os

class ModelType(Enum): 
    DPT_LARGE = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    DPT_Hybrid = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    MIDAS_SMALL = "MiDaS_small" # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

class Midas():
    def __init__(self, modelType:ModelType=ModelType.DPT_LARGE):
            
        # self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.midas = torch.hub.load("isl-org/MiDaS", "DPT_Large")
        print(f'Loaded model')
        self.modelType = modelType
        self.THRESH = 0
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using CUDA')
        else:
            print('Using CPU')
            self.device = torch.device('cpu')
    
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.modelType.value == "DPT_Large":
            self.transform = midas_transforms.dpt_transform
            self.THRESH = 26
        elif self.modelType.value == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
            self.THRESH = 1900
        else:
            self.transform = midas_transforms.small_transform
            self.THRESH = 850
        print('Loaded Transform')
        
        
    def predict(self, frame):
        t1 = time.time()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        t2 = time.time()
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depthMap = prediction.cpu().numpy()
        t3 = time.time()
        print(t3-t2, t2-t1)
        return depthMap

image_path = r"/run/media/bhavesh/Shared/backup/pathpal/depthmap/test_image2.jpg"
image = cv2.imread(image_path)
# print(image)
cv2.imshow('original image', image)

model = Midas()



dmap = model.predict(image)
depthMap = cv2.normalize(dmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depthMap_rgb = cv2.applyColorMap(depthMap, cv2.COLORMAP_INFERNO)
cv2.imshow('depthmap', dmap)