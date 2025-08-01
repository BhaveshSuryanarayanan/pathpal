import midas_module
from midas_module import Midas
import importlib

importlib.reload(midas_module)
midas_model = Midas('m')
import cv2

# image_path = r"/run/media/bhavesh/Shared/backup/pathpal/depthmap/test_image2.jpg"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    midas_model.predict(frame)
    midas_model.locate_obstacles()
    mat = midas_model.find_matrix()
    midas_model.display()
    print(mat)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        