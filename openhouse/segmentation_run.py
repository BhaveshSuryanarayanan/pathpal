import cv2
from segmentation_module import Segmentation
import time
cap = cv2.VideoCapture(0)

rn_model = Segmentation()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    t1 = time.time()
    rn_model.predict(frame)
    rn_model.display(frame)
    t2 = time.time()
    
    print(f'{int(1000*(t2-t1))} ms  FPS = {int(1/(t2-t1))}')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        