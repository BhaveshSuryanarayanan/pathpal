import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open the camera")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read the frame")
        break

    cv.imshow('frame', frame)

    # Wait for 1 millisecond, and break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
