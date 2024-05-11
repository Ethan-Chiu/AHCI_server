import socket
import cv2
from multiprocessing import Process, Pipe, Queue
import time
import websockets
import asyncio
import threading

async def ws_connect(stop_event, queue, filename):
    print("connecting to ws posedata")
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/posedata"
    websocket = await websockets.connect(websocket_uri)
    print("connect to ws posedata")
    file = open(filename, 'w')
    while not stop_event.is_set():
        message_str = await websocket.recv()
        if not queue.empty():
            queue.get()
            file.write(message_str)
    print("ws_connect ended")

def ws_connect_sync(stop_event, queue, filename):
    asyncio.run(ws_connect(stop_event, queue, filename))

def client_runner(queue=None, stop_event=None):
    w = 640
    h = 360
    fps = 10

    print("start connecting camera")
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    print("camera connected")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    filename = 'output.txt'
    asyncqueue = Queue()
    
    # connect to ws
    asyncer = threading.Thread(target=ws_connect_sync, args=(stop_event, asyncqueue, filename))
    asyncer.start()

    print("iteration started")
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
        out.write(resized)
        asyncqueue.put_nowait(0)

        time.sleep(1/fps)
        if stop_event.is_set():
            print("exiting")
            cap.release()
            out.release()
            exit()

    

if __name__ == "__main__":
    client_runner()