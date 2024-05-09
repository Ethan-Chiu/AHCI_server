import asyncio
import json
import numpy as np
import socket
from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from aiortc.contrib.signaling import BYE, object_from_string, object_to_string
import websockets
from array_video_track import ArrayVideoStreamTrack
from typing import List
import threading

import os
import sys
import time
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(grandparent_dir)

from yolov9.segment.predict import run as predict
import multiprocessing as mp
import threading
from record_hand import client_runner


async def run(pc: RTCPeerConnection, player, tracks: List[MediaStreamTrack], recorder, websocket_uri):

    def add_tracks():
        for track in tracks:
            pc.addTrack(track)

        # if player and player.audio:
        #    pc.addTrack(player.audio)

        # if player and player.video:
        #    pc.addTrack(player.video)

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

    videoFrame = ArrayVideoStreamTrack()

    terminate = False

    # Generate video data
    data_bgr = np.zeros((240, 320, 4), np.uint8)

    def generate_frame(queue):
        print("generator started")
        while not terminate:
            array = queue.get()
            print("new")
            alpha_channel = np.ones((array.shape[0], array.shape[1], 1), dtype=np.uint8) * 255
            array = np.concatenate((array, alpha_channel), axis=2)
            if type(array) == type(None):
                break
            videoFrame.set_frame(array)

    queue = mp.Queue()
    stop_event = mp.Event()

    # process_segment = mp.Process(target=predict, kwargs={'source': '0', 'queue': queue, 'stop': stop_event})
    process_segment = threading.Thread(target=client_runner, kwargs={'queue': queue, 'stop_event': stop_event})
    process_segment.start()
    thread_gen = threading.Thread(target=generate_frame, kwargs={'queue': queue})
    thread_gen.start()    

    # time.sleep(20)
    # run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            run(
                pc=pc,
                player=player,
                tracks=[videoFrame],
                recorder=recorder,
                websocket_uri=websocket_uri,
            )
        )
    except KeyboardInterrupt:
        print("Ctrl C exit")
    finally:
        # cleanup
        terminate = True
        stop_event.set()
        queue.put(None)
        print("3")
        thread_gen.join()
        print("4")
        process_segment.join()
        print("5")
        loop.run_until_complete(recorder.stop())
        print("6")
        loop.run_until_complete(pc.close())
        print("7")