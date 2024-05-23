import socket
import cv2
from multiprocessing import Queue
import time
import websockets
import asyncio
import threading

async def ws_connect(stop_event, queue):
    print("Connecting to ws posedata")
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/posedata"
    async with websockets.connect(websocket_uri) as websocket:
        print("Connected to ws posedata")
        while not stop_event.is_set():
            message_str = await websocket.recv()
            if not queue.empty():
                queue.get_nowait()  # Remove the old message
            queue.put(message_str)
    print("ws_connect ended")

def ws_connect_sync(stop_event, queue):
    asyncio.run(ws_connect(stop_event, queue))

def client_runner(queue=None, stop_event=None):
    w = 640
    h = 360
    fps = 30

    print("Start connecting camera")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera connected")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    filename = 'output.txt'
    
    # Connect to WebSocket
    asyncer = threading.Thread(target=ws_connect_sync, args=(stop_event, queue))
    asyncer.start()

    print("Iteration started")
    index = 0
    latest_message = ""

    with open(filename, 'w') as file:
        while True:
            start_time = time.time()

            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and write the frame to the output video
            resized = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            out.write(resized)

            # Get the latest message from the queue if available
            if not queue.empty():
                latest_message = queue.get()

            # Write the latest message to the file
            file.write(latest_message + "\n")

            print(f"Frame {index}: {latest_message}")  # For debug purposes, you can write to a file instead
            index += 1

            # Calculate elapsed time and sleep for the remaining frame period
            elapsed_time = time.time() - start_time
            time_to_sleep = max(0, (1 / fps) - elapsed_time)
            time.sleep(time_to_sleep)

            # Check if stop event is set
            if stop_event.is_set():
                print("Exiting")
                cap.release()
                out.release()
                break

if __name__ == "__main__":
    stop_event = threading.Event()
    message_queue = Queue()
    client_runner(queue=message_queue, stop_event=stop_event)

