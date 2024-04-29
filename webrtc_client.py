import asyncio
import json
import numpy
import socket
from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from aiortc.contrib.signaling import BYE, object_from_string, object_to_string
import websockets
from array_video_track import ArrayVideoStreamTrack
from typing import List
import threading




async def run(pc: RTCPeerConnection, player, tracks: List[MediaStreamTrack], recorder, websocket_uri):

    def add_tracks():
        for track in tracks:
            pc.addTrack(track)

        if player and player.audio:
            pc.addTrack(player.audio)

        if player and player.video:
            pc.addTrack(player.video)

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

    data_bgr = numpy.zeros((240, 320, 4), numpy.uint8)
    terminate = False

    def generate_frame():
        i = 0
        while not terminate:
            data_bgr[:, :] = (i % 255, 0, 0, 0)
            i += 1
            videoFrame.set_frame(data_bgr)

    thread = threading.Thread(target=generate_frame)
    thread.start()

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
        thread.join()
        loop.run_until_complete(recorder.stop())
        loop.run_until_complete(pc.close())