import midas_module
from midas_module import Midas
import importlib
import cv2
from mqtt_module import MQTT
import time
import numpy as np

midas_model = Midas()

broker = "test.mosquitto.org"
port = 1883
send_topic = "vibrator/matrix"
receive_topic = "vibrator/switch_state"

mqtt_obj = MQTT(broker = broker, port = port, send_topic=send_topic, receive_topic=receive_topic)
mqtt_obj.initialize_connection()

matrix = np.zeros((3, 3), dtype=int)


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
    
    mqtt_obj.send_matrix(mat)
    print(mqtt_obj.switch_states)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        