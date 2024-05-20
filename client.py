import socket
import numpy as np
import cv2
import sys
from multiprocessing import Process, Pipe
import time
import io
import os
from PIL import Image
import websockets
import asyncio
import threading

async def ws_connect(stop_event, pipe):
    print("connecting to ws posedata")
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/posedata"
    websocket = await websockets.connect(websocket_uri)
    print("connect to ws posedata")
    while not stop_event.is_set():
        message_str = await websocket.recv()
        if pipe.poll():
            pipe.recv()
            pipe.send(message_str)
    print("ws_connect ended")

def ws_connect_sync(stop_event, pipe):
    asyncio.run(ws_connect(stop_event, pipe))

def send_images(server_address, port, pipe):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_address, port))
    print("connected")
    try:
        while True:
            image_bytes = pipe.recv()
            print("received")
            if image_bytes is None:
                break
            client_socket.sendall(image_bytes)
            client_socket.send(b"IMAGE_COMPLETE")
            print("sent")
            print(b'handhead' in image_bytes)

            result_bytes = b''
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                result_bytes += chunk
                if result_bytes.endswith(b"ARRAY_COMPLETE"):
                    result_bytes = result_bytes[:-14]
                    break
            print("received 2")
            if result_bytes:
                image = Image.open(io.BytesIO(result_bytes))
                result = np.array(image)[:, :, :3]
                pipe.send(result)
                print("sent 2")
    finally:
        client_socket.close()


def displayer(pipes, main, fps, queue):
    print("displayer started")
    index = 0
    while True:
        result = pipes[0].recv()
        print("displaying")
        if queue:
            queue.put(result)
            continue
        cv2.imshow(f'Image', result)
        index += 1
        if cv2.waitKey(int(1000/fps*0.9)) & 0xFF == ord('q'):
            main.send(None)
            break
    cv2.destroyAllWindows()

def client_runner(queue=None, stop_event=None):
    w = 640
    h = 360
    fps = 15

    print("start connecting camera")
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    print("camera connected")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    server_address = "140.112.30.57"
    base_port = 13752
    
    # connect to ws
    # asyncio .run(ws_connect(stop_event))
    asyncpipe = Pipe()
    asyncer = Process(target=ws_connect_sync, args=(stop_event, asyncpipe[1]))
    asyncer.start()

    pipes = Pipe()
    displipe = Pipe()

    processes = []
    process = Process(target=send_images, args=(server_address, base_port, pipes[1]))
    processes.append(process)
    process.start()
    
    display = Process(target=displayer, args=(pipes, displipe[1], fps, queue))
    display.start()

    print("iteration started")
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        asyncpipe[0].send(0)
        resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
        image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
        handhead = "handhead" + asyncpipe[0].recv()
        return_bytes = image_bytes + bytes(handhead, 'utf-8')
        pipes[0].send(return_bytes)
        if displipe[0].poll(1/fps):
            if displipe[0].recv() is None:
                break
        index += 1
        if stop_event.is_set():
            print("exxxxit")
            exit()

    for pipe in pipes:
        pipe[0].send(None)

    time.sleep(1)
    print("start exiting")
    display.join()
    print("display joined")
    for pipe in pipes:
        if pipe[0].poll():
            pipe[0].recv()
            print("pipe joined")
    for process in processes:
        process.join()
        print("process joined")

if __name__ == "__main__":
    client_runner()