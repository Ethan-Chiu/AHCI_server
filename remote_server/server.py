import socket
import numpy as np
import subprocess
import io
import sys
import threading
import time

# Function to process the image using predict.py
def process_image(image_bytes, process):
    # Send the image data to the subprocess
    image_bytes += b"X" * (1024 - (len(image_bytes) % 1024))
    process.stdin.write(image_bytes)
    process.stdin.flush()

    output = b''
    while True:
        chunk = process.stdout.read(1024)
        if not chunk:
            break
        output += chunk
        if output.endswith(b"X") and b'ARRAY_COMPLETE' in output:  # Check if the completion signal is received
            output = output.split(b'ARRAY_COMPLETE')[0] + b'ARRAY_COMPLETE'
            break
    return output

server_address = '140.112.30.57'
server_address = '127.0.0.1'
server_sockets, client_sockets, processes, threads = [], [], [], []
devices = [0]
for i in range(int(sys.argv[1])):
    # Create a TCP/IP socket for the first server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    port = int(sys.argv[2]) + i
    # Bind the socket to the server address for the first server
    server_socket.bind((server_address, port))

    # Listen for incoming connections for the first server
    server_socket.listen(1)

    server_sockets.append(server_socket)
    print(f"Server {i} is listening on", port)

    process = subprocess.Popen(['python', 'predict.py', str(devices[i])], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    processes.append(process)

def handle_connection(client_socket, process):
    try:
        while True:
            # Receive the image data from the client
            byte_data = b''
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                byte_data += chunk
                if byte_data.endswith(b"IMAGE_COMPLETE"):  # Check if the completion signal is received
                    break

            if not byte_data:
                break

            # print(time.time())

            # Process the received image
            print("received")
            result = process_image(byte_data, process)

            # Send the result back to the client
            client_socket.sendall(result)
            print("sent")

    finally:
        # Close the connection
        client_socket.close()

try:
    # Accept connections for both servers
    while True:
        print("Waiting for connections...")
        for i, server_socket in enumerate(server_sockets):
            client_socket, client_address = server_socket.accept()
            print("Connection established from", client_address)
            client_sockets.append(client_socket)

            thread = threading.Thread(target=handle_connection, args=(client_sockets[-1], processes[i]))
            thread.start()
            threads.append(thread)
except KeyboardInterrupt:
    print("Server stopped.")
except Exception as e:
    print("E:",e)
finally:
    for server_socket in server_sockets:
        server_socket.close()

