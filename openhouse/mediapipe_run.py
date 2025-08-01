from hand_tracking_module import HandTracking
import numpy as np
import cv2
import time


mp_model = HandTracking()

cap = cv2.VideoCapture(0)

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    groceries = mp_model.predict(frame)
    mp_model.display(frame, show=True)
    
    t2 = time.time()
    
    print(f'{int(1000*(t2-t1))}  FPS = {int(1/(t2-t1))}')
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cv2.waitKey(0)
cv2.destroyAllWindows()

