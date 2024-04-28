import math
import cv2
import numpy
from aiortc import (
    RTCIceCandidate,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

class ArrayVideoStreamTrack(VideoStreamTrack):

    def __init__(self, queue):
        super().__init__()  # don't forget this!
        self.counter = 0
        height, width = 480, 640

        # generate data here (an array with shape (height, width, 3))
        self.data_bgr = numpy.zeros((height, width, 4), numpy.uint8)

        # shrink and center it
        # M = numpy.float32([[0.5, 0, width / 4], [0, 0.5, height / 4]])
        # data_bgr = cv2.warpAffine(data_bgr, M, (width, height))

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
