import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import copy

grocery_model = YOLO(r'/home/bhavesh/pathpal/models/grocery_picking/yolov8x.pt')

cap = cv2.VideoCapture(0)
# cap.set(10,200)

def draw_grid(image):
    HEIGHT,WIDTH,CHANNELS = image.shape
    one_third_height = HEIGHT//3
    one_third_width = WIDTH//3
    new_image = np.copy(image)
    cv2.line(new_image,(one_third_width,0),(one_third_width,HEIGHT),(0,0,0),5)
    cv2.line(new_image,(2*one_third_width,0),(2*one_third_width,HEIGHT),(0,0,0),5)
    cv2.line(new_image,(0,one_third_height),(WIDTH,one_third_height),(0,0,0),5)
    cv2.line(new_image,(0,2*one_third_height),(WIDTH,2*one_third_height),(0,0,0),5)
    return new_image

def vibration_matrix(vib_matrix):
    WIDTH, HEIGHT = 400, 400
    one_third_height = HEIGHT//3
    one_third_width = WIDTH//3
     
    output = np.zeros((HEIGHT,WIDTH,3))
    
    for i in range(3):
        for j in range(3):
            startx = one_third_width*i
            endx = startx + one_third_width
            starty = one_third_height*j
            endy = starty + one_third_height
            
            clr = (0,255,0) if vib_matrix[j][i]==0 else (0,0,255)
            cv2.rectangle(output,(startx,starty),(endx,endy),clr,thickness = cv2.FILLED)
            
    output = draw_grid(output)
    
    return output


while True:
    ret, frame = cap.read()
    # cv2.imshow('og image', frame)
    
    results = grocery_model.predict(source=frame, save=False, imgsz=640, conf=0.40, show=False, show_labels=True, save_txt=False)
    temp = copy.deepcopy(frame)
    
    object_centre = 0
    
    target = ['bread','book']
    # print(results[0].names)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            
            # if not grocery_model.names[cls].lower() in target:
            #     continue
            
            conf = box.conf.item()
            xyxy = box.xyxy.flatten().tolist()

            detected_objects.append({
                'class':grocery_model.names[cls],
                'confidence': conf,
                'bounding_box': xyxy
            })
            
            x1, y1, x2, y2 = xyxy
            object_centre = ((x1+x2)//2,(y1+y2)//2)
            
    annotated_image = results[0].plot()
    if detected_objects:
        print(detected_objects)
    else:
        print('No grocery detected')
        # annotated_image = temp
    
    # cv2.imshow('annotated image', annotated_image)  
    # continue
    
    ''' HAND TRACKING '''
    
    def calculateFingers(res, drawing):
        # Ensure the contour has enough points
        if len(res) > 3:
            hull = cv2.convexHull(res, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(res, hull)
                
                if defects is not None:
                    cnt = 0
                    for i in range(defects.shape[0]):  # calculate the angle
                        s, e, f, d = defects[i][0]
                        start = tuple(res[s][0])
                        end = tuple(res[e][0])
                        far = tuple(res[f][0])
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                            cnt += 1
                            cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                    if cnt > 0:
                        return True, cnt + 1
                    else:
                        return True, 0
        return False, 0

    def get_center_of_contour(contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            return None

    # Open Camera
    # camera = cv2.VideoCapture(0)
    # camera.set(10, 200)
    bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
    st = time.time()
    
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing
    # frame = cv2.flip(frame, 1)  # Horizontal Flip
    # cv2.imshow('original', frame)
    
    fgmask = bgModel.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    img = cv2.bitwise_and(frame, frame, mask=fgmask)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow('Threshold Hands', skinMask)
    
    skinMask1 = copy.deepcopy(skinMask)
    contours, _ = cv2.findContours(skinMask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = -1
    res = None
    if len(contours) > 0:
        for i in range(len(contours)):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                res = temp
        if res is not None and len(res) > 3:
            hull = cv2.convexHull(res)
            drawing = annotated_image
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            # Calculate the center of the hand contour
            center = get_center_of_contour(res)
            if center:
                cv2.circle(drawing, center, 10, (255, 0, 0), -1)  # Draw center
                # print("Center coordinates:", center)
            
            cv2.namedWindow('output', cv2.WINDOW_NORMAL)  # Allows resizing
            cv2.resizeWindow('output', 800, 600)   
            cv2.imshow('output', drawing)
    

    if object_centre==0: 
        object_centre = (100,100)
        center = (100,100)
        # print('no')
    xob, yob = object_centre
    xhand, yhand = center
    
    xrel = xob - xhand
    yrel = yob - yhand
    
    h = 100
    w = 100
    
    xvib, yvib = 0,0
    
    # print(xrel,yrel)
     
    if xrel > w/2:
        xvib = 2
    elif xrel > -w/2:
        xvib = 1
    else :    
        xvib = 0
    
    if yrel > h/2:
        yvib = 2
    elif yrel > -h/2:
        yvib = 1
    else :    
        yvib = 0
    
    
    vib = [[0 for i in range(3)] for j in range(3)]
    vib[yvib][xvib]=1
    
    # print(vib)
    
    vib_output = vibration_matrix(vib)
    # cv2.imshow('Coin vibrators',vib_output)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       
    