from yolo_module import Yolo
from hand_tracking_module import HandTracking
from grocery_picking_module import Grocery_picking 
import cv2
from mqtt_module import MQTT

cap = cv2.VideoCapture(0)
gp_model = Grocery_picking('94.9_model.pt') 

broker = "test.mosquitto.org"
port = 1883
send_topic = "vibrator/matrix"
receive_topic = "vibrator/switch_state"
mqtt_obj = MQTT(broker = broker, port = port, send_topic=send_topic, receive_topic=receive_topic)
mqtt_obj.initialize_connection()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    mat = gp_model.predict(frame)
    gp_model.display(frame)
    mqtt_obj.send_matrix(mat, show=False)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
cv2.waitKey(0)
cv2.destroyAllWindows()
