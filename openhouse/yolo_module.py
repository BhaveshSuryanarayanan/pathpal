import ultralytics
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import seaborn as sns

class Yolo():
    
    def __init__(self, model_path="yolo11x.pt"):
        self.model = YOLO(model_path)
        print(self.model.names)
        self.create_color_pallete()
    
    def create_color_pallete(self):
        self.color_pallete = [(int(r * 255), int(g * 255), int(b * 255)) 
                            for r, g, b in sns.color_palette("hsv", len(self.model.names))]

    def predict(self, frame):
        self.frame = frame
        results = self.model.predict(source=frame, save=False, imgsz=640, conf=0.40, show=False, show_labels=False, save_txt=False, verbose=False)
        
        '''
        The output is stored as a list of dictionaries, Organization is as follows
        Dectected object = [object0, object1, ...]
        Objecti = {
            'class' : the class of the object
            'confidence' : Confidence of prediction (ones with less confidence can be removed by setting a threshold)
            'bounding_box' : [x1,y1,x2,y2]
            where (x1,y1) and (x2,y2) are the diagnol vertices of bounding box
        }
        '''
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                conf = box.conf.item()
                xyxy = box.xyxy.flatten().tolist()

                detected_objects.append({
                    'class_num' : cls,
                    'class': self.model.names[cls],
                    'confidence': conf,
                    'bounding_box': xyxy
                })
                
        
        self.detected_objects = detected_objects
        return detected_objects
        
    def display(self, frame, show=True):
        original_image = frame.copy()
        annotator = Annotator(original_image, example=self.model.names)
        
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['bounding_box']
            annotator.box_label([x1, y1, x2, y2], label = f"{obj['class']} {obj['confidence']:.2f}", color=self.color_pallete[obj['class_num']])
        
        annotated_image = annotator.result()
        
        if show:
            cv2.imshow('Annotated Image', annotated_image)
        
        return annotated_image