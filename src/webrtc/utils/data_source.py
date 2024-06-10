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
        self.task = None
        self.logger = get_logger("PoseDataSource")

    async def start(self):
        self.logger.info("Starting...")
        try: 
            connected = await self.__connect_ws_server()
            if not connected:
                return False
            
            self.task = asyncio.create_task(self.__receive_data())
            return self.connected
        except: 
            self.logger.error("Failed to start")
    
    async def end(self):
        self.logger.info("Ending...")
        if self.task:
            self.task.cancel()
        if self.websocket:
            await self.websocket.close()
        self.logger.info("Pose data source closed")

    async def get_data(self):
        try: 
            if not self.queue.empty():
                latest_data = self.queue.get_nowait()
            else:
                latest_data = b''  # Default value if no data is available
            return latest_data
        except Exception as e: 
            self.logger.error(f"Failed to get data: {e}")
            raise Exception("Get pose data error")

    async def __receive_data(self):
        try:
            while True:
                data = await self.websocket.recv()
                await self.queue.put(data)
        except Exception as e:
            self.logger.error(f"WebSocket receive error: {e}")

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
    def __init__(self, cam_source: int = 1):
        self.cap = None
        self.cam_source = cam_source
        self.logger = get_logger("CameraDataSource")

    async def start(self):
        self.logger.info("Starting...")
        try:
            self.logger.info("Start connecting camera")
            self.cap = cv2.VideoCapture(self.cam_source, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.logger.info("Camera connected")
            return True
        except:
            self.logger.error("Failed to start camera")
            return False

    def get_data(self):
        try:
            ret, frame = self.cap.read()
            self.logger.debug(str(frame.shape))
            if not ret:
                self.logger.error("No camera input!")
                return None
            resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
            image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
            return image_bytes, frame
        except Exception as e:
            self.logger.error(f"Failed to get data: {e}")
            raise Exception("Get camera data error")
    
    def end(self):
        self.logger.info("Camera data source closed")
        return
