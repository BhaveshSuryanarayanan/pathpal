import numpy as np

import time
import cv2 
import torch 
from enum import Enum
import os

from pp_utils import *


class ModelType(Enum): 
    DPT_LARGE = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    DPT_Hybrid = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    MIDAS_SMALL = "MiDaS_small" # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)


class Midas(): 
    def __init__(self, model_type = 'l'):
        
        if model_type=='m':
            modelType = ModelType.DPT_Hybrid
        elif model_type == 's':
            modelType = ModelType.MIDAS_SMALL
        else:
            modelType = ModelType.DPT_LARGE
            
        # self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)   # load specified model
        print(f'Loaded model {modelType.value}')
        
        self.modelType = modelType
        self.THRESH = 0
        
        # initialize device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using CUDA')
        else:
            print('Using CPU')
            self.device = torch.device('cpu')

        self.midas.to(self.device)
        self.midas.eval()
        
        self.load_transform()
        self.set_threshold()
            
    # load transform for image preprocessing
    def load_transform(self):
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.modelType.value == "DPT_Large":
            self.transform = midas_transforms.dpt_transform
        elif self.modelType.value == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        print('Loaded Transform')
    
    # function to set obstacle detection threshold
    def set_threshold(self):
        
        '''
        adjust distance threshold as required
        '''
        
        if self.modelType.value == "DPT_Large":
            self.DIST_THRESH = 26
        elif self.modelType.value == "DPT_Hybrid":
            self.DIST_THRESH = 1900
        else:
            self.DIST_THRESH = 850
    
    # run prediction on given image
    def predict(self, frame):
        t1 = time.time()
        
        # preprocessing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        # predict
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        self.depthmap = prediction.cpu().numpy()
        
        t2 = time.time()
        print('inference time : ', int(1000*(t2-t1)), ' ms')
        
        self.frame = frame
        
        # return the depthmap
        return self.depthmap
        

    def locate_obstacles(self, overlay=True):
        '''
        applies threshold to identify the obstacles
        
        Parameters:
            overlay = True - the mask is superimposed on the image (for display purposes)
        '''
        
        self.overlay = overlay
        
        # self.DIST_THRESH, norm_depthmap = self.find_anchoring_threshold(self.depthmap)
        # self.mask = norm_depthmap > self.DIST_THRESH  # generate mask using the Distance threshold
        self.mask = self.depthmap >self.DIST_THRESH
        # overlaying the mask on the original image
        if overlay == True:
            self.mask_on_image = self.frame.copy()

            self.mask_on_image[self.mask] = [255, 0, 84] 
            self.mask_on_image = draw_grid_on_image(self.mask_on_image)
        
        return self.mask
    
    def find_anchoring_threshold(self, depthmap):
        norm_depthmap = cv2.normalize(depthmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        mean = np.mean(norm_depthmap[0:450, 560:640])
        print(norm_depthmap.shape)
        thresh = mean*(6/25)**0.30
        norm_depthmap[0:450, 560:640] = 0
        
        return thresh, norm_depthmap
    
    def find_matrix(self):
        """
        From the provided Mask, it returns the coin vibrator matrix by applying a 
        percentage threshold on each cell
        """
        
        # Dimensions of the mas
        HEIGHT,WIDTH = self.mask.shape
        one_third_height = HEIGHT//3
        one_third_width = WIDTH//3
        
        self.matrix = np.zeros((3,3), dtype=np.int16) # initialize matrix as zero
        
        self.PERCENTAGE_THRESHOLD = 20  # set percentage threshold
        
        # check each box
        for i in range(3):
            for j in range(3):
                startx = one_third_width*i
                endx = startx + one_third_width
                starty = one_third_height*j
                endy = starty + one_third_height
                
                box = self.mask[starty:endy, startx:endx] # obtain the mask within the box
                
                h, w = box.shape 
                size = h*w
                
                
                self.matrix[j][2-i] = np.sum(box==1) >= size*self.PERCENTAGE_THRESHOLD/100  # if obstacle cover more than percentage treshold then assign 1
        
        return self.matrix # return coin vibrator matrix
    
    
    # displaying the output (demo purposes)
    def display(self, show=True):
        '''
        Images displayed
        1. depthmap (colormap)
        2. mask over image (if overlay==True)
        3. vibration matrix
        '''
        
        # set size of image, make sure it doesn't go out of screen
        dm_height, dm_width = self.depthmap.shape
        if dm_width<=600:
            width, height = dm_width, dm_height
            dmap = self.depthmap
        else:
            width = 600
            height = int(dm_height*(width/dm_width))
            dmap = cv2.resize(self.depthmap, (width, height))
            
        # draw the vibration matrix (it must have a square shape)
        self.vibmat = VibrationMatrix.draw_vibration_matrix(self.matrix, (height, height), 'round')
        
        # normalize the depth and convert to rgb(to support opencv)
        normalized_depthMap = cv2.normalize(dmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.depthMap_rgb = cv2.applyColorMap(normalized_depthMap, cv2.COLORMAP_INFERNO)
        
        seg1 = cv2.flip(self.depthMap_rgb,1)
        seg3 = self.vibmat
        
        # combine the three/ two segments using hconcat
        display_image = seg1.copy()
        if self.overlay:
            seg2 = cv2.flip(cv2.resize(self.mask_on_image, (width, height)), 1)
            display_image = cv2.hconcat([display_image, seg2])
        display_image = cv2.hconcat([display_image, seg3])
        
        if show:
            cv2.imshow("Depthmaping", display_image)
        
        return display_image
        