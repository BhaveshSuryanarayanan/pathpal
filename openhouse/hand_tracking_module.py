import mediapipe as mp
import cv2
import numpy as np

class HandTracking():
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        # print("Using GPU:", self.hands._use_gpu)  # Should print True if GPU is used
        
    def predict(self, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result_hands = self.hands.process(self.frame)
        
        if not self.result_hands.multi_hand_landmarks:
            print('No hands detected')
            return None
        
        for hand_landmarks in self.result_hands.multi_hand_landmarks:
            self.center = self.get_hand_center(hand_landmarks.landmark, frame.shape)
        
        return self.center
        
    def get_hand_center(self, landmarks, image_shape):
        height, width, _ = image_shape
        x_sum, y_sum = 0, 0
        for lm in landmarks:
            x_sum += lm.x
            y_sum += lm.y
        cx = int((x_sum / len(landmarks)) * width)
        cy = int((y_sum / len(landmarks)) * height)
        return (cx, cy)
        
    def display(self, image, full_hand = True, show=True):    
        output = image.copy()  
        mp_draw = mp.solutions.drawing_utils
        
        if not self.result_hands.multi_hand_landmarks:
            if show:
                cv2.imshow("Hand Tracking", output)
            return output
        
        if full_hand:
            for hand_landmarks in self.result_hands.multi_hand_landmarks:
                mp_draw.draw_landmarks(output, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        cv2.circle(output, self.center, 10, (0, 255, 0), -1)
        
        if show:
            cv2.imshow("Hand Tracking", output)
        
        self.output = output
        return output