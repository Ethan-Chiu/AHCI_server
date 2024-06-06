import socket
import multiprocessing as mp
import time
import random
import websockets
import asyncio
import cv2
import numpy as np
import time
import logging

from logger.logger_util import get_logger

class PoseDataSource:
    def __init__(self):
        self.websocket = None
        self.connected = False
        self.queue = asyncio.Queue()
        self.logger = get_logger("PoseDataSource")

    async def start(self):
        self.logger.info("Starting...")
        try: 
            connected = await self.__connect_ws_server()
            if not connected:
                return False
            
            asyncio.create_task(self.__receive_data())
            return self.connected
        except: 
            self.logger.error("Failed to start")
    
    def end(self):
        self.logger.info("Ending...")
        if self.websocket:
            self.websocket.close()
        self.logger.info("Pose data source closed")

    async def __receive_data(self):
        try:
            while True:
                data = await self.websocket.recv()
                await self.queue.put(data)
        except Exception as e:
            self.logger.error(f"WebSocket receive error: {e}")
    
    async def get_data(self):
        if not self.queue.empty():
            latest_data = self.queue.get_nowait()
        else:
            latest_data = b''  # Default value if no data is available
        return latest_data

    async def __connect_ws_server(self):
        self.logger.info("Connecting to ws posedata")
        try:
            host = socket.gethostbyname(socket.gethostname())
            websocket_uri = f"ws://{host}:8080/posedata"
            self.websocket = await websockets.connect(websocket_uri)
            self.connected = True
            self.logger.info("Connected to ws posedata")
        except:
            self.logger.error("Error connecting to ws server posedata")
        return self.connected


class CameraDataSource:
    def __init__(self):
        self.cap = None
        self.logger = get_logger("CameraDataSource")

    async def start(self):
        self.logger.info("Starting...")
        try:
            self.logger.info("Start connecting camera")
            self.cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.logger.info("Camera connected")
            return True
        except:
            self.logger.error("Failed to start camera")
            return False

    async def get_data(self):
        ret, frame = self.cap.read()
        self.logger.debug(str(frame.shape))
        if not ret:
            self.logger.error("No camera input!")
            return None
        resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
        image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
        return image_bytes, frame
    
    def end(self):
        self.logger.info("Camera data source closed")
        return

class ProducerConsumer:
    def __init__(self, logger: logging.Logger, latest_data, lock, stop_event, out_pipe, cam_queue, fps):
        self.latest_data = latest_data
        self.lock = lock
        self.stop_event = stop_event
        self.out_pipe = out_pipe
        self.cam_queue = cam_queue
        self.fps = fps
        self.logger = logger

        self.pose_data_source = PoseDataSource()
        self.cam_data_source = CameraDataSource()

    async def start(self):
        self.logger.info("Starting...")
        try: 
            self.logger.info("Starting data sources")
            pose_data_init, cam_data_init = await asyncio.gather(
                self.pose_data_source.start(),
                self.cam_data_source.start()
            )
            return pose_data_init and cam_data_init
        except: 
            self.logger.error("Failed to start")
            return False

    async def run(self):
        try:
            i = 0
            start_time = time.time()
            print("check 0")
            while not self.stop_event.is_set():
                # Produce data
                print("check 1")
                pose_data = await self.pose_data_source.get_data()
                print("check 2")
                cam_data, frame_data = await self.cam_data_source.get_data()
                print("check 3")

                if not pose_data or not cam_data:
                    await asyncio.sleep(1)
                    continue

                handhead = "handhead" + pose_data
                return_bytes = cam_data + bytes(handhead, 'utf-8')
                print("check 6")

                self.cam_queue.put_nowait(frame_data)
                self.out_pipe.send(return_bytes)
                print("check 7")

                sleep_time = max(0, 1/self.fps - (time.time() - start_time))
                time.sleep(sleep_time)
                print("check 8")
                start_time = time.time()
                print("check 9")

        except asyncio.exceptions.CancelledError:
            print("Run cancelled")
        except Exception as err:
            print(f"Run error: Unexpected {err=}, {type(err)=}")
        finally:
            self.pose_data_source.end()
            self.cam_data_source.end()
            print("Run stopped")

    def close(self):
        self.pose_data_source.end()
        self.cam_data_source.end()
        while self.out_pipe.poll():
            self.out_pipe.recv()


class Distributor:
    def __init__(self, producerconsumer_args):
        self.latest_data = [b"", b"", np.ones((640, 480, 3)), -1]  # Use a list to store the latest data (mutable type)
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.logger = get_logger("Distributor")

        self.producerconsumer = ProducerConsumer(self.logger, self.latest_data, self.lock, self.stop_event, **producerconsumer_args)

        self.producerconsumer_task = None

    async def start(self):
        self.logger.info("Starting...")
        try:
            return await self.__start()
        except asyncio.exceptions.CancelledError:
            self.logger.warn("Distributor cancelled")
        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
        return False
    
    def run(self):
        asyncio.run(self.__run())

    async def __start(self):
        producerconsumer_init = await self.producerconsumer.start()
        if not producerconsumer_init:
            self.logger.error("Failed to init ProducerConsumer")
            return False
        self.logger.info("ProducerConsumer initialized")
        return True

    async def __run(self):
        # Start long running task
        self.producerconsumer_task = asyncio.create_task(self.producerconsumer.run())
        await self.producerconsumer_task

    def stop(self):
        self.logger.info("Stopping...")
        self.stop_event.set()
        self.producerconsumer.close()

        if self.producerconsumer_task:
            self.producerconsumer_task.cancel()
        self.logger.info("Distributor stopped")