import socket
import cv2
import struct
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
print(len(encoded_image),encoded_image.shape, encoded_image[10]) #53731


# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket')
# Connect to the server
client_socket.connect((SERVER_IP, PORT))
print(f'Connected to {SERVER_IP}:{PORT}')
print('connected')

if not cap.isOpened():
    print("Error: Could not open webcam.")
    client_socket.close()
    exit()

if ret:
    # Display the captured frame
    cv2.imshow('Captured Image', frame)
    # print("Press any key to send the image.")
    # cv2.waitKey(0)
    print(frame.shape)
    # Encode the frame as JPEG to send it over the socket
    # result, encoded_image = cv2.imencode('.jpg', frame)
    print('sending')
    print(encoded_image.shape, encoded_image[10]) #53731

    if result:
        # Send the encoded image data over the socket
        size = len(encoded_image)
    
        # Send the size of the image packed in a 4-byte integer
        client_socket.sendall(struct.pack(">L", size))  # ">L" ensures big-endian 4-byte unsigned int
        
        client_socket.sendall(encoded_image.tobytes())
        print('Image sent successfully.')

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Close the socket connection
client_socket.close()
