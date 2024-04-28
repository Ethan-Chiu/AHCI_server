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

    def __init__(self):
        super().__init__()  # don't forget this!
        self.counter = 0
        height, width = 480, 640

        # generate data here (an array with shape (height, width, 3))
        self.data_bgr = numpy.hstack(
            [
                self._create_rectangle(
                    width=213, height=480, color=(255, 0, 0)
                ),  # blue
                self._create_rectangle(
                    width=214, height=480, color=(255, 255, 255)
                ),  # white
                self._create_rectangle(width=213, height=480, color=(0, 0, 255)),  # red
            ]
        )

        # shrink and center it
        # M = numpy.float32([[0.5, 0, width / 4], [0, 0.5, height / 4]])
        # data_bgr = cv2.warpAffine(data_bgr, M, (width, height))

    def _convertArrayToVideoFrame(self, array):
        return VideoFrame.from_ndarray(array, format="bgr24")
    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame = self._convertArrayToVideoFrame(self.data_bgr)
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1
        return frame

    def _create_rectangle(self, width, height, color):
        data_bgr = numpy.zeros((height, width, 3), numpy.uint8)
        data_bgr[:, :] = color
        return data_bgr
