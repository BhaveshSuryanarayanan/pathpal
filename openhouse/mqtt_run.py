from mqtt_module import MQTT
import time
import numpy as np

broker = "test.mosquitto.org"
port = 1883
send_topic = "vibrator/matrix"
receive_topic = "vibrator/switch_state"

mqtt_obj = MQTT(broker = broker, port = port, send_topic=send_topic, receive_topic=receive_topic)
mqtt_obj.initialize_connection()

matrix = np.zeros((3, 3), dtype=int)

while True:
    for i in range(3):
        for j in range(3):
            print(mqtt_obj.switch_states)
            matrix[i][j] = 1
            s= mqtt_obj.switch_states
            print(s)
            
            mqtt_obj.send_matrix(matrix)
            time.sleep(2)
            matrix[i][j] = 0 