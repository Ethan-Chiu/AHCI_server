import asyncio
import json
import cv2
import socket
from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, add_signaling_arguments, create_signaling, object_from_string, object_to_string
import websockets
from array_video_track import ArrayVideoStreamTrack

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(grandparent_dir)

from yolov9.segment.predict import run as predict
import multiprocessing as mp
import threading
import numpy as np


async def run(pc: RTCPeerConnection, player, recorder, websocket_uri, role):
    videoFrame = ArrayVideoStreamTrack()
    def add_tracks():
        pc.addTrack(videoFrame)

        # if player and player.audio:
        #     pc.addTrack(player.audio)

        # if player and player.video:
        #     pc.addTrack(player.video)

    # connect signaling

    def generate_frame(queue, stop):
        while not stop.is_set():
            array = queue.get()
            print(array)
            videoFrame.set_frame(array)

    websocket = await websockets.connect(websocket_uri)

    if role == "offer":
        add_tracks()
        await pc.setLocalDescription(await pc.createOffer())

        data = object_to_string(pc.localDescription)
        await websocket.send(json.dumps({
            "Type": "OFFER",
            "Message": data,
        }))

    # videoFrame.set_frame()
    queue = mp.Queue()
    stop_event = mp.Event()
    child1 = mp.Process(target=predict, kwargs={'source': 'http://172.20.10.4/mjpeg/1', 'queue': queue, 'stop': stop_event})
    child1.start()
    child2 = threading.Thread(target=generate_frame, kwargs={'queue': queue, 'stop': stop_event})
    child2.start()    
    
    # # consume signaling
    while True:
        message_str = await websocket.recv()
        print(message_str)
        message = json.loads(message_str)
        type = message["Type"]
        data = message["Message"]

        obj = object_from_string(data)

        if type == "answer" and isinstance(obj, RTCSessionDescription):
            print("Get answer!")
            await pc.setRemoteDescription(obj)
            await recorder.start()

        elif type == "candidate" and isinstance(obj, RTCIceCandidate):
            print("Add candidate!")
            await pc.addIceCandidate(obj)

        elif obj is BYE:
            print("Exiting")
            stop_event.set()
            child1.join()
            child2.join()
            break

    

if __name__ == "__main__":
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/video"  # Replace with your WebSocket server URI

    # peer connection
    pc = RTCPeerConnection()

    # create media source
    player = MediaPlayer("./test.mp4")
    recorder = MediaBlackhole()

    # run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            run(
                pc=pc,
                player=player,
                recorder=recorder,
                websocket_uri=websocket_uri,
                role="offer",
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        # cleanup
        loop.run_until_complete(recorder.stop())
        loop.run_until_complete(pc.close())