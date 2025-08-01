import ultralytics
from ultralytics import YOLO
import cv2 as cv

# Loads the model
model = YOLO("94.9_model.pt")

cap = cv.VideoCapture(0)
# predicts output for given image 
# show = True displays the image

while True:
    ret, frame = cap.read()
    
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.40, show=True, show_labels=True, save_txt=False)

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
                'class': model.names[cls],
                'confidence': conf,
                'bounding_box': xyxy
            })

    print(detected_objects)

# import cv2
# x = input()
# cv2.destroyAllWindows()