from yolo_module import Yolo
from hand_tracking_module import HandTracking
from grocery_picking_module import Grocery_picking 
import cv2
from segmentation_module import Segmentation

# gp_model = Yolo()

image_path = r'/run/media/bhavesh/Shared/backup/pathpal/road diversion images/IMG20240816123533.jpg'
img = cv2.imread(image_path)
# gp_model.predict(img)
# gp_model.display()

rn_model = Segmentation()
rn_model.predict(img)
rn_model.display(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
