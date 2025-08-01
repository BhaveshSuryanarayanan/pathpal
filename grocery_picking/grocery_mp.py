import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
import copy
#i = 0
cap = cv2.VideoCapture(0)
#time.sleep(5)
#i += 1
# Load YOLO model
# grocery_model = YOLO(r'C:\Users\yagni\PythonProjects\Pathpal\SWITCH\best_93.pt')
grocery_model = YOLO('yolov8x.pt')
# Initialize the webcam
#cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def draw_grid(image):
    HEIGHT, WIDTH, CHANNELS = image.shape
    one_third_height = HEIGHT // 3
    one_third_width = WIDTH // 3
    new_image = np.copy(image)
    cv2.line(new_image, (one_third_width, 0), (one_third_width, HEIGHT), (0, 0, 0), 5)
    cv2.line(new_image, (2 * one_third_width, 0), (2 * one_third_width, HEIGHT), (0, 0, 0), 5)
    cv2.line(new_image, (0, one_third_height), (WIDTH, one_third_height), (0, 0, 0), 5)
    cv2.line(new_image, (0, 2 * one_third_height), (WIDTH, 2 * one_third_height), (0, 0, 0), 5)
    return new_image

def vibration_matrix(vib_matrix):
    WIDTH, HEIGHT = 400, 400
    one_third_height = HEIGHT // 3
    one_third_width = WIDTH // 3
    
    output = np.zeros((HEIGHT, WIDTH, 3))
    
    for i in range(3):
        for j in range(3):
            startx = one_third_width * i
            endx = startx + one_third_width
            starty = one_third_height * j
            endy = starty + one_third_height
            
            clr = (0, 255, 0) if vib_matrix[j][i] == 0 else (0, 0, 255)
            cv2.rectangle(output, (startx, starty), (endx, endy), clr, thickness=cv2.FILLED)
            
    output = draw_grid(output)
    
    return output

def get_hand_center(landmarks, image_shape):
    height, width, _ = image_shape
    x_sum, y_sum = 0, 0
    for lm in landmarks:
        x_sum += lm.x
        y_sum += lm.y
    cx = int((x_sum / len(landmarks)) * width)
    cy = int((y_sum / len(landmarks)) * height)
    return (cx, cy)

while True:
    ret, frame = cap.read()
    st = time.time()

    # YOLO object detection
    # results = grocery_model.predict(source=frame, save=False, imgsz=640, conf=0.40, show=False, show_labels=True, save_txt=False)
    results = grocery_model.predict(source=frame, save=False, imgsz=640, conf=0.40, show=False, show_labels=True, save_txt=False, classes=[24,39,46,47,49,50,51,73])
                                    
    temp = copy.deepcopy(frame)
    
    object_centre = None
    detected_objects = []
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = box.conf.item()
            xyxy = box.xyxy.flatten().tolist()
            detected_objects.append({
                'class': grocery_model.names[cls],
                'confidence': conf,
                'bounding_box': xyxy
            })
            x1, y1, x2, y2 = xyxy
            object_centre = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Draw bounding box around detected objects
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, f"{grocery_model.names[cls]}: {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # if detected_objects:
        # print(detected_objects)
    # else:
        # print('No grocery detected')

    # MediaPipe Hand Detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(frame_rgb)

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            center = get_hand_center(hand_landmarks.landmark, frame.shape)

            # Only draw the center of the hand
            cv2.circle(frame, center, 10, (255, 0, 0), -1)  # Draw center
            # print("Center coordinates:", center)

            # Relative position between object center and hand center
            if object_centre:
                xob, yob = object_centre
                xhand, yhand = center
                xrel = xob - xhand
                yrel = yob - yhand

                h = 100
                w = 100

                xvib, yvib = 0, 0

                if xrel > w / 2:
                    xvib = 2
                elif xrel > -w / 2:
                    xvib = 1
                else:
                    xvib = 0

                if yrel > h / 2:
                    yvib = 2
                elif yrel > -h / 2:
                    yvib = 1
                else:
                    yvib = 0

                vib = [[0 for _ in range(3)] for _ in range(3)]
                vib[yvib][xvib] = 1
                vib_output = vibration_matrix(vib)
                cv2.imshow('Coin vibrators', vib_output)
    end = time.time()
    # Show the processed frame
    cv2.imshow('Annotated Image', frame)
    # print(st-end)
    # Exit on 'q'
    #if i==15:
    #    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
