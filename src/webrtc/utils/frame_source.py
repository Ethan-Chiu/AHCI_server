import threading
import numpy as np 
import multiprocessing as mp
from .array_video_track import ArrayVideoStreamTrack

class FrameSource:
    def __init__(self, queue: mp.Queue):
        self.videoFrame = ArrayVideoStreamTrack()
        self.queue = queue
        self.thread = None
        self.stop_event = mp.Event()

    def _generate_frame(self):
        print("Generator started")
        while not self.stop_event.is_set():
            array = self.queue.get()
            if type(array) == type(None):
                print("Get None")
                break
            print("Get new frame")
            alpha_channel = np.ones((array.shape[0], array.shape[1], 1), dtype=np.uint8) * 255
            array = np.concatenate((array, alpha_channel), axis=2)
            self.videoFrame.set_frame(array)

    def listen(self):
        self.thread = threading.Thread(target=self._generate_frame)
        self.thread.start()

    def stop(self):
        print("Stopping FrameSource")
        self.stop_event.set()
        self.thread.join()
        print("FrameSource stopped")

    def get_source_track(self):
        return self.videoFrame