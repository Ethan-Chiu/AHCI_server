import socket
import cv2
import time
import websockets
import asyncio
import os

async def ws_connect(stop_event, queue):
    print("Connecting to ws posedata")
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/posedata"
    async with websockets.connect(websocket_uri) as websocket:
        print("Connected to ws posedata")
        i = 0
        while not stop_event.is_set():
            message_str = await websocket.recv()
            if not queue.empty():
                await queue.get()  # Remove the old message
            await queue.put(message_str)
            print(f"get {i}")
            i += 1
    print("ws_connect ended")

async def capture_video(queue, stop_event):
    w = 640
    h = 360
    fps = 30
    base_path = 'C:/Users/david/Desktop/AHCI/data/common'
    base_folder_name = 'commond'
    existing_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    
    # 計算新的資料夾名稱
    folder_count = sum([1 for folder in existing_folders if folder.startswith(base_folder_name)])
    new_folder_name = f"{base_folder_name}_{folder_count + 1}"
    target_folder = os.path.join(base_path, new_folder_name)
    print("Start connecting camera")
    os.makedirs(target_folder)
    videopath = os.path.join(target_folder, f'{new_folder_name}.avi')
    filename = os.path.join(target_folder, f'{new_folder_name}.txt')
    print(videopath, filename)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera connected")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(videopath, fourcc, fps, (w, h))

    print("Iteration started")
    index = 0
    latest_message = ""

    with open(filename, 'w') as file:
        while not stop_event.is_set():
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
                latest_message = await queue.get()

            # Write the latest message to the file
            file.write(latest_message)

            print(f"Frame {index}: {latest_message}")  # For debug purposes, you can write to a file instead
            index += 1

            # Calculate elapsed time and sleep for the remaining frame period
            elapsed_time = time.time() - start_time
            time_to_sleep = max(0, (1 / fps) - elapsed_time)
            await asyncio.sleep(time_to_sleep)

    print("Exiting")
    cap.release()
    out.release()

async def main():
    stop_event = asyncio.Event()
    message_queue = asyncio.Queue()

    # Run WebSocket connection and video capture concurrently
    await asyncio.gather(
        ws_connect(stop_event, message_queue),
        capture_video(message_queue, stop_event)
    )

if __name__ == "__main__":
    asyncio.run(main())

