from mqtt_module import MQTT
import time
import numpy as np
import cv2
from midas_module import Midas
from grocery_picking_module import Grocery_picking 
import grocery_recognition 

broker = "test.mosquitto.org"
port = 1883
send_topic = "vibrator/matrix"
receive_topic = "vibrator/switch_state"
mqtt_obj = MQTT(broker = broker, port = port, send_topic=send_topic, receive_topic=receive_topic)
mqtt_obj.initialize_connection()
class_constraint = None
midas_model = Midas('m')
gp_model = Grocery_picking(model_path=r'/run/media/bhavesh/Shared/backup/pathpal/openhouse/94.9_model.pt') 

cap = cv2.VideoCapture(0)

while True:
    ret, frame  = cap.read()
    
    if not ret:
        print('Failed to grab frame')
        break
    
    s0, s1, s2 = mqtt_obj.switch_states
    
    print(s0, s1, s2)
    
    if s0==0 and s1==0 and s2==1:
        switch = 1
    elif s0 == 0 and s1 == 0 and s2 == 0:
        switch = 3
    else:
        switch = 2
        
    
    if switch==1:
        midas_model.predict(frame)
        midas_model.locate_obstacles()
        mat = midas_model.find_matrix()
        midas_model.display()
        # print(mat, 'dmap')
        mqtt_obj.send_matrix(mat, show=False)

    elif switch == 3:
        # global class_constraint
        class_constraint = grocery_recognition.get_grocery_list()
        time.sleep(1)
        
    elif switch==2:
        gp_model.class_constraint = class_constraint
        flipped_frame = cv2.flip(frame, 1)
        mat = gp_model.predict(flipped_frame)
        gp_model.display(flipped_frame)
        # print(mat, 'g')
        # print()
        # print(mat, 'gp')
        mqtt_obj.send_matrix(mat, show=True)
    # else:
    #     class_constraint = grocery_recognition.get_grocery_list()

            
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()