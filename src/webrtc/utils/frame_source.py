import threading
import numpy as np 
import multiprocessing as mp
from .array_video_track import ArrayVideoStreamTrack

from logger.logger_util import get_logger

class FrameSource:
    def __init__(self, queue: mp.Queue, name: str):
        self.videoFrame = ArrayVideoStreamTrack()
        self.queue = queue
        self.thread = None
        self.stop_event = mp.Event()
        self.logger = get_logger(f"WebRTC FrameSource {name}")

    def _generate_frame(self):
        self.logger.info("Generator started")
        while not self.stop_event.is_set():
            array = self.queue.get()
            if array is None:
                self.logger.info("Generator stopped")
                break
            self.logger.debug("Get new frame")
            alpha_channel = np.ones((array.shape[0], array.shape[1], 1), dtype=np.uint8) * 255
            array = np.concatenate((array, alpha_channel), axis=2)
            self.videoFrame.set_frame(array)

    def listen(self):
        self.thread = threading.Thread(target=self._generate_frame)
        self.thread.start()

    def stop(self):
        self.logger.warning("Stopping FrameSource")
        self.stop_event.set()
        self.queue.put(None)
        self.thread.join()
        self.logger.info("FrameSource stopped")

    def get_source_track(self):
        return self.videoFrame