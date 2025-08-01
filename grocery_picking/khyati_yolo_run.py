
import ultralytics
from ultralytics import YOLO

# Loads the model
model = YOLO("khyati_yolo_model.pt")

# predicts output for given image 
# show = True displays the image
FILEPATH = ""
results = model.predict(source=FILEPATH, save=False, imgsz=640, conf=0.40, show=True, show_labels=True, save_txt=False)

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

print(results[0].names)
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

'''
{0: 'Apple', 1: 'Banana', 2: 'Book', 3: 'Carrots', 4: 'Cerealbox', 5: 'Detergent',
6: 'Drinks', 7: 'Egg', 8: 'Lemon', 9: 'Meat', 10: 'Milk', 11: 'Orange', 12: 'Tomato',
13: 'Watermelon', 14: 'bread'}
'''