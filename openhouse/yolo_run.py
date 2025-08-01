from yolo_module import Yolo
import cv2
import time
yolo_model = Yolo("94.9_model.pt")
# yolo_model = Yolo("yolo11x.pt")

cap = cv2.VideoCapture(0)

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    flipped_frame = cv2.flip(frame, 1)
    groceries = yolo_model.predict(flipped_frame)
    yolo_model.display(flipped_frame, show=True)
    
    t2 = time.time()
    
    print(f'{int(1000*(t2-t1))}  FPS = {int(1/(t2-t1))}')
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cv2.waitKey(0)
cv2.destroyAllWindows()



