import os
import sys
import asyncio
import json
import socket
import threading
import websockets
import multiprocessing as mp
from typing import List

from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from aiortc.contrib.signaling import BYE, object_from_string, object_to_string

from utils.frame_source import FrameSource
from utils.yolo_runner_client import YoloRunnerClient



async def run(pc: RTCPeerConnection, tracks: List[MediaStreamTrack], websocket_uri):

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

    stop_event = mp.Event()

    # Queue for video
    queue = mp.Queue()

    # Get video from queue
    videoSource = FrameSource(queue)
    videoSource.listen()

    # Send camera to server
    # Put result in queue
    # TODO: send pose data to server
    yoloClient = YoloRunnerClient(queue, server_ip="140.112.30.57", server_num=1)
    yoloClient.connect()
    yoloClient.display()


    # Connect to Unity by WebRTC
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            run(
                pc=pc,
                tracks=[videoSource.get_source_track()],
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
        print("Closing peer connection")
        loop.run_until_complete(pc.close())
        print("Peer connection closed")