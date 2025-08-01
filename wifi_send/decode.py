import socket
import cv2
import numpy as np

# Server IP and port (replace with the IP address of the server)
# SERVER_IP = '192.168.190.157' # Change to the actual server IP
SERVER_IP = '10.42.66.36'
PORT = 12345


# Open the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture('asmitha.png')


# Capture a frame from the webcam
ret, frame = cap.read()
frame = cv2.resize(frame, (255,255))

print(frame.shape)#,frame[10][245:250])

result, encoded_image = cv2.imencode('.jpg', frame)
print(encoded_image.shape,encoded_image[:10])

send = encoded_image.tobytes()
receive = np.frombuffer(send, dtype=np.uint8)

np.savetxt("send_data.txt",receive,fmt='%d')
print(receive[:10])
decoded_image = cv2.imdecode(receive, cv2.IMREAD_COLOR)
    
# Check the shape and a pixel value
# print("Decoded image shape:", decoded_image.shape)

# Optionally, display the decoded image using OpenCV
cv2.imshow("Decoded Image", decoded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

