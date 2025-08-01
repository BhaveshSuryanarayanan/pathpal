import time
import paho.mqtt.client as mqtt

class MQTT():
    
    def __init__(self, broker = "test.mosquitto.org", port = 1883, send_topic = "vibrator/matrix", receive_topic = "vibrator/switch_state"):
        
        # setup variables
        self.broker = broker    # mqtt-broker
        self.port = port    # communication port
        self.send_topic = send_topic    # topic for coin vibrators
        self.receive_topic = receive_topic  # topic of switch
        
        self.switch_states = [0, 0, 0]      
        
    def initialize_connection(self):
        # start mqtt connection
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port)
        self.client.subscribe(self.receive_topic)
        self.client.loop_start() # start connection
        
        print("MQTT connection initiated")
    
    '''def get_switch_state(self):
        self.client.loop_read()
        return self.switch_states'''
    
    def on_message(self,client, userdata, message):
        # receive the switch states from the esp32 as a string
        switch_state_str = message.payload.decode()
        print("Received switch states : ", switch_state_str)
        
        # store switch states in an array
        self.switch_states = [int(switch_state_str[0]), int(switch_state_str[2]), int(switch_state_str[4])]
        
        
    def send_matrix(self, matrix, show=True):
        # print(type(matrix))
        # send vibration matrix as a string
        matrix_str = "\n".join([" ".join(map(str, row)) for row in matrix])
        
        self.client.publish(self.send_topic, matrix_str) # publish
        if show:
            print("Published Matrix :\n", matrix_str)
        
    # end mqtt connection
    def disconnected(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected MQTT")
            
    
        