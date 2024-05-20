import socket
import numpy as np
import cv2
import sys
from multiprocessing import Process, Pipe
import time
import io
from PIL import Image

def send_images(server_address, port, pipe):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, port))
    try:
        while True:
            image_bytes = pipe.recv()
            if image_bytes is None:
                break
            client_socket.sendall(image_bytes)
            client_socket.send(b"IMAGE_COMPLETE")

            result_bytes = b''
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                result_bytes += chunk
                if result_bytes.endswith(b"ARRAY_COMPLETE"):
                    result_bytes = result_bytes[:-14]
                    break
            image = Image.open(io.BytesIO(result_bytes))
            result = np.array(image)[:, :, :3]
            pipe.send(result)
    finally:
        client_socket.close()

def displayer(pipes, main, fps, servers, queue):
    index = 0
    while True:
        result = pipes[index % servers][0].recv()
        if queue:
            queue.put(result)
            continue
        cv2.imshow(f'Image {index}', result)
        index += 1
        if cv2.waitKey(int(1000/fps*0.9)) & 0xFF == ord('q'):
            main.send(None)
            break
    cv2.destroyAllWindows()

def client_runner(queue=None, stop_event=None):
    cap = cv2.VideoCapture(1)
    server_address = "140.112.30.57"
    servers = 1
    base_port = 13751
    fps = 15

    pipes = [Pipe() for _ in range(servers)]
    displipe = Pipe()

    processes = []
    for i in range(servers):
        port = base_port + i
        process = Process(target=send_images, args=(server_address, port, pipes[i][1]))
        processes.append(process)
        process.start()

    display = Process(target=displayer, args=(pipes, displipe[1], fps, servers, queue))
    display.start()

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
        image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
        pipes[index % servers][0].send(image_bytes)
        if displipe[0].poll(1/fps):
            if displipe[0].recv() is None:
                break
        index += 1

    for pipe in pipes:
        pipe[0].send(None)

    time.sleep(1)
    display.join()
    for pipe in pipes:
        if pipe[0].poll():
            pipe[0].recv()
    for process in processes:
        process.join()

if __name__ == "__main__":
    client_runner()