import asyncio
import json
import socket
from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from aiortc.contrib.signaling import BYE, object_from_string, object_to_string
import websockets
from typing import List
import threading

import os
import sys
import time
import cv2

from utils.frame_source import FrameSource
from utils.yolo_runner_client import YoloRunnerClient

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(grandparent_dir)

import multiprocessing as mp
import threading
from record.record_hand import client_runner



async def run(pc: RTCPeerConnection, tracks: List[MediaStreamTrack], recorder, websocket_uri):

    def add_tracks():
        for track in tracks:
            pc.addTrack(track)

    # connect signaling
    websocket = await websockets.connect(websocket_uri)

    # add tracks
    add_tracks()

    # create offer
    await pc.setLocalDescription(await pc.createOffer())
    data = object_to_string(pc.localDescription)
    await websocket.send(json.dumps({
        "Type": "OFFER",
        "Message": data,
    }))
    
    # consume signaling
    while True:
        message_str = await websocket.recv()
        print(message_str)
        message = json.loads(message_str)
        type = message["Type"]
        data = message["Message"]

        obj = object_from_string(data)
        print("while")

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
            break
    

if __name__ == "__main__":
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/video"  # Replace with your WebSocket server URI

    # peer connection
    pc = RTCPeerConnection()

    # create media source
    player = MediaPlayer("./test.mp4")
    recorder = MediaBlackhole()

    stop_event = mp.Event()

    queue = mp.Queue()

    videoSource = FrameSource(queue)
    videoSource.listen()

    yoloClient = YoloRunnerClient(queue, server_ip="140.112.30.57", server_num=1)
    yoloClient.connect()

    # time.sleep(20)
    # run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            run(
                pc=pc,
                tracks=[videoSource.get_source_track()],
                recorder=recorder,
                websocket_uri=websocket_uri,
            )
        )
    except KeyboardInterrupt:
        print("Ctrl C exit")
    finally:
        # cleanup
        stop_event.set()
        queue.put(None)
        videoSource.stop()
        yoloClient.close()
        print("5")
        loop.run_until_complete(recorder.stop())
        print("6")
        loop.run_until_complete(pc.close())
        print("7")