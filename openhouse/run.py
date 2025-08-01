from mqtt_module import MQTT
import time
import numpy as np
import cv2
from midas_module import Midas
from grocery_picking_module import Grocery_picking 
import argparse

parser = argparse.ArgumentParser(description="Control output display")
parser.add_argument("--display", "-d", action="store_true", help="Display output")
args = parser.parse_args()

broker = "test.mosquitto.org"
port = 1883
send_topic = "vibrator/matrix"
receive_topic = "vibrator/switch_state"

midas_model = Midas('m')
gp_model = Grocery_picking('94.9_model.pt') 

mqtt_obj = MQTT(broker = broker, port = port, send_topic=send_topic, receive_topic=receive_topic)
mqtt_obj.initialize_connection()

cap = cv2.VideoCapture(0)

while True:
    ret, frame  = cap.read()
    
    if not ret:
        print('Failed to grab frame')
        break
    
    gp_model.class_constraint = ['apple', 'orange', 'strawberry', 'drinks']
    s0, s1, s2 = mqtt_obj.switch_states
    print(s0, s1, s2)
    if s0==0 and s2==0:
        switch = 1
    else:
        switch = 2
        
    if switch==1:
        midas_model.predict(frame)
        midas_model.locate_obstacles()
        mat = midas_model.find_matrix()
        
        if args.display:
            midas_model.display()
        mqtt_obj.send_matrix(mat, show=False)
        
    elif switch==2:
        mat = gp_model.predict(frame)
        
        if args.display:
            gp_model.display(frame)

        mqtt_obj.send_matrix(mat, show=True)
            
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()