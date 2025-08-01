from unet_model import UNET, transform
import matplotlib.pyplot as plt
import torch 
from PIL import Image
import cv2
import os
import numpy as np

class Segmentation():
    def __init__(self, relative_path = "segmentation_torch_unet.pth"): 
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, "segmentation_torch_unet.pth")

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using CUDA for segmentation')
        else:
            print('Using CPU for segmentation')
            self.device = torch.device('cpu')
            
        self.model = UNET().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path,map_location=torch.device(self.device),weights_only=True))

        self.model.eval()

    def predict(self, img):
        if isinstance(img, np.ndarray):  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = Image.fromarray(img) 
        final_img = transform(img)

        final_img = final_img.unsqueeze(0).to(self.device)

        prediction = self.model(final_img)

        output = prediction < 0.5  
        output = output.squeeze(0).squeeze(0).cpu().numpy() 

        self.mask = output
        
        return self.mask


    def display(self, img, show=True):
        window_size = 600
        img_height, img_width = img.shape[:2]  
        
        if img_height>window_size:
            img=cv2.resize(img, ( int((window_size*img_width)/img_height),window_size))
            
        img_height, img_width = img.shape[:2]  

        mask_resized = cv2.resize(self.mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        mask_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        mask_color[:, :, 2] = (mask_resized * 255)  
        
        alpha = 1  
        overlay = cv2.addWeighted(img, 1, mask_color, alpha, 0)

        if show:
            window_name = "Road Segmentation"
            cv2.imshow(window_name, overlay)

        return overlay