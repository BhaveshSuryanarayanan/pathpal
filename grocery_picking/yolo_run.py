import torch
import cv2

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_93.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)

    # Display the results
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv5', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
    