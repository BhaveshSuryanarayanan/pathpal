from hand_tracking_module import HandTracking
import cv2

# gp_model = Yolo()

# image_path = r'/run/media/bhavesh/Shared/backup/pathpal/grocery_picking/test_image_5.jpg'
# img = cv2.imread(image_path)
# gp_model.predict(img)
# gp_model.display()
cap = cv2.VideoCapture(0)

mp_model = HandTracking()   

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    print(mp_model.predict(frame))
    mp_model.display(frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cv2.waitKey(0)
cv2.destroyAllWindows()
