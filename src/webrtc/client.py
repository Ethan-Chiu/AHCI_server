import os
import sys
import asyncio
import json
import socket
import threading
import websockets
import multiprocessing as mp
from typing import List
import logging

from logger.logger_util import get_logger

from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from aiortc.contrib.signaling import BYE, object_from_string, object_to_string

from utils.frame_source import FrameSource
from utils.yolo_runner_client import YoloRunnerClient

async def run(pc: RTCPeerConnection, tracks: List[MediaStreamTrack], websocket):

    def add_tracks():
        for track in tracks:
            pc.addTrack(track)
            print(track.id)

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
    try:
        while True:
            message_str = await websocket.recv()
            print(message_str)
            message = json.loads(message_str)
            msg_type = message["Type"]
            data = message["Message"]

            obj = object_from_string(data)
            print("while")

            if msg_type == "answer" and isinstance(obj, RTCSessionDescription):
                print("Get answer!")
                await pc.setRemoteDescription(obj)

            elif msg_type == "candidate" and isinstance(obj, RTCIceCandidate):
                print("Add candidate!")
                await pc.addIceCandidate(obj)

            elif obj is BYE:
                print("Exiting")
                break
    except asyncio.CancelledError:
        print("Signaling task cancelled")
    

async def main():
    host = socket.gethostbyname(socket.gethostname())
    websocket_uri = f"ws://{host}:8080/video"  # Replace with your WebSocket server URI

    # peer connection
    pc = RTCPeerConnection()

    # Queue for video
    queue = mp.Queue()
    cam_queue = mp.Queue()

    # Get video from queue
    videoSource = FrameSource(queue, name="Video")
    videoSource.listen()

    camSource = FrameSource(cam_queue, name="Camera")
    camSource.listen()

    # Send camera to server
    # Put result in queue
    # "140.112.30.57"
    yoloClient = YoloRunnerClient(queue, cam_queue, server_ip="127.0.0.1", port=12345)
    connect = asyncio.create_task(yoloClient.connect())
    display = asyncio.create_task(yoloClient.display())
    connect_init, display_init = await asyncio.gather(connect, display)

    async def cleanup():
        yoloClient.close()
        camSource.stop()
        videoSource.stop()
        logger.info("Closing peer connection")
        await pc.close()
        logger.info("Peer connection closed")

    if not connect_init or not display_init:
        logger.warning(f"Failed to init. Connection: {connect_init}, Display: {display_init}")
        await cleanup()
        return
    
    # Connect to Unity by WebRTC
    try:
        # connect signaling
        async with websockets.connect(websocket_uri) as websocket:
            await run(
                pc=pc,
                tracks=[camSource.get_source_track(), videoSource.get_source_track()],
                websocket=websocket,
            )
    except asyncio.CancelledError:
        logger.warning("Main task cancelled")
    except Exception as e:
        logger.error(f"Error in main task: {e}")
    finally:
        await cleanup()

def main_wrapper(logger: logging.Logger):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
        logger.info("Completed")
    except KeyboardInterrupt:
        logger.warning("Ctrl c exiting...")
    finally:
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    logger = get_logger("WebRTC Client")
    main_wrapper(logger)
