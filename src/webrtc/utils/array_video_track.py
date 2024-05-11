import numpy
from aiortc import (
    RTCIceCandidate,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

class ArrayVideoStreamTrack(VideoStreamTrack):

    def __init__(self):
        super().__init__()  # don't forget this!
        self.counter = 0
        height, width = 480, 640

        # generate data here (an array with shape (height, width, 3))
        self.data_bgr = numpy.zeros((height, width, 4), numpy.uint8)

    def set_frame(self, frame):
        self.data_bgr = frame

    def _convertArrayToVideoFrame(self, array):
        return VideoFrame.from_ndarray(array, format="bgra")
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self._convertArrayToVideoFrame(self.data_bgr)
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1
        return frame
