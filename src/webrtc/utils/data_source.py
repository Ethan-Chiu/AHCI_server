import socket
import multiprocessing as mp
import time
import random
import websockets
import asyncio
import cv2


class PoseDataSource:
    def __init__(self):
        self.websocket = None
        self.connected = False

    async def start(self):
        await self.__connect_ws_server()
        return self.connected
    
    async def get_data(self):
        print("Getting data")
        return await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
    
    def end(self):
        pass 

    async def __connect_ws_server(self):
        print("Connecting to ws posedata")
        try:
            host = socket.gethostbyname(socket.gethostname())
            websocket_uri = f"ws://{host}:8080/posedata"
            self.websocket = await websockets.connect(websocket_uri)
            self.connected = True
            print("Connected to ws posedata")
        except:
            print("Error connecting to ws server posedata")


class CameraDataSource:
    def __init__(self):
        self.cap = None

    async def start(self):
        try:
            print("Start connecting camera")
            self.cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
            print("Camera connected")
            return True
        except:
            print("Failed to connect to camera")
            return False

    async def get_data(self):
        ret, frame = self.cap.read()
        if not ret:
            print("No camera input!")
            return None
        resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
        image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
        return image_bytes
    
    def end(self):
        return


class Producer:
    def __init__(self, latest_data, lock, stop_event):
        self.latest_data = latest_data
        self.lock = lock
        self.stop_event = stop_event

        self.pose_data_source = PoseDataSource()
        self.cam_data_source = CameraDataSource()

    async def start(self):
        print("Starting data sources")
        pose_data_init, cam_data_init = await asyncio.gather(
            self.pose_data_source.start(),
            self.cam_data_source.start()
        )
        return pose_data_init and cam_data_init

    async def produce(self):
        try:
            while not self.stop_event.is_set():
                pose_data = await self.pose_data_source.get_data()
                cam_data = await self.cam_data_source.get_data()
                # NOTE: testing
                # cam_data = b"test"
                async with self.lock:
                    self.latest_data[0] = pose_data
                    self.latest_data[1] = cam_data
                print(f"Produced data!")
        except asyncio.exceptions.CancelledError:
            print("Producer cancelled")
        finally:
            self.pose_data_source.end()
            self.cam_data_source.end()


class Consumer:
    def __init__(self, latest_data, lock, out_pipe, stop_event):
        self.latest_data = latest_data
        self.lock = lock
        self.stop_event = stop_event
        self.out_pipe = out_pipe

    async def consume(self):
        try:
            while not self.stop_event.is_set():
                async with self.lock:
                    pose_data = self.latest_data[0]
                    cam_data = self.latest_data[1]

                if(pose_data == b"" or cam_data == b""):
                    print("Data empty")
                    await asyncio.sleep(1)
                    continue

                print(f"Consumed data!")
                handhead = "handhead" + pose_data
                return_bytes = cam_data + bytes(handhead, 'utf-8')
                
                print("sending to remote server...")
                self.out_pipe.send(return_bytes)
                print("sent to remote server")

                await asyncio.sleep(1)
                # TODO: gather to increase performance
                # await asyncio.gather(
                #     asyncio.to_thread(),
                #     asyncio.sleep(1)
                # )
        except asyncio.exceptions.CancelledError:
            print("Consumer cancelled")
        finally:
            print("Consumer stopped")


class Distributor:
    def __init__(self, out_pipe):
        self.latest_data = [b"", b""]  # Use a list to store the latest data (mutable type)
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.out_pipe = out_pipe

        self.producer = Producer(self.latest_data, self.lock, self.stop_event)
        self.consumer = Consumer(self.latest_data, self.lock, out_pipe, self.stop_event)

        self.producer_task = None
        self.consumer_task = None

    def start(self):
        print("Distributor starting")
        try:
            asyncio.run(self.__start())
        except asyncio.exceptions.CancelledError:
            print("Distributor cancelled")

    async def __start(self):
        producer_init = await self.producer.start()
        if not producer_init:
            print("Failed to init producer")
            return
        print("Producer initialized")

        # Start long running task
        self.producer_task = asyncio.create_task(self.producer.produce())
        self.consumer_task = asyncio.create_task(self.consumer.consume())
        await asyncio.gather(self.producer_task, self.consumer_task)

    def stop(self):
        print("Stopping producer")
        self.stop_event.set()
        if self.producer_task:
            self.producer_task.cancel()
        if self.consumer_task:
            self.consumer_task.cancel()
        print("Producer stopped")