import socket

# Define server host and port
HOST = '0.0.0.0'  # Listen on all available interfaces (Wi-Fi)
PORT = 12345  # Choose any available port

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((HOST, PORT))

# Listen for incoming connections (queue up to 5 connections)
server_socket.listen(1)
print(f'Server listening on {HOST}:{PORT}...')

# Accept incoming connection
client_socket, client_address = server_socket.accept()
print(f'Connected by {client_address}')

# Receive data from the client
while True:
    data = client_socket.recv(1024)  # Receive up to 1024 bytes
    if not data:
        break  # No more data, close the connection
    print(f"Received from client: {data.decode('utf-8')}")

# Close the connections
client_socket.close()
server_socket.close()