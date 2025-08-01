from yolo_module import Yolo
from hand_tracking_module import HandTracking
import numpy as np
import cv2
from pp_utils import *

class Grocery_picking():
    
    def __init__(self, model_path = "yolo11x.pt", class_constraint = None):
        self.yolo_model = Yolo(model_path)
        self.mp_model = HandTracking()
        self.class_constraint = class_constraint
        self.hand_found = True
        self.grocery_found = True
      
    def predict(self, frame):
        self.frame = frame
        self.hand_center = self.mp_model.predict(frame)
        self.groceries = self.yolo_model.predict(frame)
        
        self.vib_matrix = np.zeros((3,3), dtype=np.uint)
        self.hand_found=True
        self.grocery_found = True
        
        if self.hand_center==None:
            self.hand_found = False
        if self.groceries==None:
            self.grocery_found = False
            print('No groceries detected')
        
        if not (self.hand_found and self.grocery_found):
            return np.zeros((3,3),dtype=np.int16)
        
        selected_grocery = self.find_nearest_grocery()
        if not selected_grocery:
            print('Grocery not found in list')
            self.grocery_found = False
            return np.zeros((3,3), dtype = np.int16)
        
        self.vib_matrix = self.find_matrix(self.hand_center, selected_grocery)
        return self.vib_matrix
        
        
    def distance(self, c1, c2):
        return (c1[0]-c2[0])**2+(c1[1]-c2[1])**2
        
    # def find_nearest_grocery(self):
    #     min_distance = float('inf')
    #     min_grocery = None
        
    #     for grocery in self.groceries:  
    #         if self.class_constraint!=None and grocery not in self.class_constraint:
    #             continue
            
    #         x1, y1, x2, y2 = grocery['bounding_box']
    #         c1 = ((x1+x2)/2, (y1+y2)/2)
    #         c2 = self.hand_center
    #         d = self.distance(c1, c2)
    #         if d < min_distance:
    #             min_grocery = grocery
    #             min_distance = d
        
    #     x1, y1, x2, y2 = grocery['bounding_box']
    #     xob, yob = int((x1+x2)//2), int((y1+y2)//2)
    #     self.grocery_center = (xob, yob)
        
    #     return grocery
       
    def find_nearest_grocery(self):
        min_distance = float('inf')
        min_grocery = None

        for grocery in self.groceries:
            # Only consider groceries in class_constraint
            if self.class_constraint is None or grocery['class'].lower() in self.class_constraint:
                x1, y1, x2, y2 = grocery['bounding_box']
                c1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                c2 = self.hand_center

                d = self.distance(c1, c2)
                if d < min_distance:
                    min_grocery = grocery.copy()
                    min_distance = d
                    self.grocery_center = (int((x1 + x2) // 2), int((y1 + y2) // 2))  # Update grocery center

        return min_grocery 
    
    def find_matrix(self, hand_center, grocery):
        H, W = 100, 100
        x1, y1, x2, y2 = grocery['bounding_box']
        xob, yob = int((x1+x2)//2), int((y1+y2)//2)
        
        xhand, yhand = hand_center
        xrel = xob - xhand
        yrel = yob - yhand
        xvib, yvib = 0, 0

        if xrel > W / 2:
            xvib = 2
        elif xrel > -W / 2:
            xvib = 1
        else:
            xvib = 0

        if yrel > H / 2:
            yvib = 2
        elif yrel > -H / 2:
            yvib = 1
        else:
            yvib = 0

        vib_matrix = np.zeros((3,3), dtype=np.int16)
        vib_matrix[yvib][2-xvib] = 1
        
        return vib_matrix
    
    def display(self, frame, show=True):
        output = frame.copy()
        
        if self.grocery_found:
            output = self.yolo_model.display(output, show=False)
        
        if self.hand_found:
            output = self.mp_model.display(output, full_hand=True, show=False)
        
        if self.grocery_found and self.hand_found:
            cv2.arrowedLine(output, 
                    self.hand_center,  # End point
                    self.grocery_center,  # Start point
                    (255, 255, 0),  # Blue color
                    10,  # Thickness
                    tipLength=0.1)  # Size of the arrowhead
        
        if show:
            dm_height, dm_width, _ = output.shape
            if dm_width<=1200:
                width, height = dm_width, dm_height
                seg1 = output
            else:
                width = 1200
                height = int(dm_height*(width/dm_width))
                seg1 = cv2.resize(output, (width, height))
            
            
            seg1 = cv2.flip(seg1,1)
            seg2 = VibrationMatrix.draw_vibration_matrix(self.vib_matrix,(height, height), "round")
            
            display_image = cv2.hconcat([seg1, seg2])
            cv2.imshow("Grocery Picking", display_image)
            
        return output