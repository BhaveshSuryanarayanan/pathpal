import socket

# Define server host and port
# SERVER_HOST = '10.42.64.83'  # Replace with server's IP address
SERVER_HOST = '192.168.41.157'  # Replace with server's IP address
SERVER_PORT = 12345  # Same as the server port

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket')
# Connect to the server
client_socket.connect((SERVER_HOST, SERVER_PORT))
print('connected')

# Send messages to the server

while True:
    message = input("Enter message to send (type 'exit' to quit): ")
    # message = 'wtf'
    if message.lower() == 'exit':
        break
    client_socket.sendall(message.encode('utf-8'))

# Close the connection
client_socket.close()
