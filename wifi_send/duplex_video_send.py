import socket
import cv2
import struct
import time
# Server IP and port (replace with the IP address of the server)
# SERVER_IP = '192.168.190.157' # Change to the actual server IP
SERVER_IP = '192.168.41.157'
PORT = 12345

# Open the webcam (0 is usually the default webcam)


# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket')
# Connect to the server
client_socket.connect((SERVER_IP, PORT))
print(f'Connected to {SERVER_IP}:{PORT}')
print('connected')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    client_socket.close()
    exit()

i = 0
while True:
    t1 = time.time()
    i+=1

    ret, frame = cap.read()
    frame = cv2.resize(frame, (255,255))
    result, encoded_image = cv2.imencode('.jpg', frame)

    if ret:
        cv2.imshow('Captured Image', frame)
        # result, encoded_image = cv2.imencode('.jpg', frame)
        print('sending')

        if result:
            # Send the encoded image data over the socket
            size = len(encoded_image)
        
            # Send the size of the image packed in a 4-byte integer
            client_socket.sendall(struct.pack(">L", size))  # ">L" ensures big-endian 4-byte unsigned int
            
            client_socket.sendall(encoded_image.tobytes())
            print('Image sent successfully.')

    response = client_socket.recv(1024).decode()
    print(f"Received from server: {response}")

    # time.sleep(0.3)

    t2 = time.time()
    print(int(1000*(t2-t1)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

client_socket.close()
