import socket

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

SERVER = '192.168.41.157'
# Connect to the server
client_socket.connect((SERVER, 12345))

# Send and receive messages
while True:
    # Sending a message to the server
    message = input("Enter message to send to server: ")
    client_socket.send(message.encode())
    
    # Receiving response from the server
    response = client_socket.recv(1024).decode()
    print(f"Received from server: {response}")

# Close the socket
client_socket.close()