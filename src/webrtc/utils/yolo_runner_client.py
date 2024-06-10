import io
import socket
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe
from PIL import Image
import threading
from .distributor import Distributor

from logger.logger_util import get_logger

class YoloRunnerClient:
    def __init__(self, queue: mp.Queue, cam_queue:mp.Queue, server_ip, port):
        self.queue = queue
        self.cam_queue = cam_queue

        self.server_address = server_ip
        self.base_port = port

        self.logger = get_logger("YoloRunnerClient")

        self.pipe_distributor, self.pipe_receiver = Pipe()
        self.display_pipe_parent, self.display_pipe = Pipe()

        self.send_image_threads: list[threading.Thread] = []
        self.display_process: threading.Thread | None = None
        self.worker = None
        self.stop_event = threading.Event()

        self.fps = 5

        producerconsumer_args = dict(
            out_pipe=self.pipe_distributor, 
            fps=self.fps,
            cam_queue=self.cam_queue
        )

        self.distributor = Distributor(producerconsumer_args)


    async def connect(self):
        distributor_init_done = threading.Event()
        distributor_error = threading.Event()

        send_image_init_done = threading.Event()

        self.worker = threading.Thread(target=self.distributor.run, args=(distributor_init_done, distributor_error))
        self.worker.start()

        distributor_init = distributor_init_done.wait(20.0)
        if not distributor_init:
            self.logger.error("Distributor init timeout")
            return False
        if distributor_error.is_set():
            self.logger.error("Distributor error")
            return False

        port = self.base_port
        process = threading.Thread(target=self.__send_images, args=(port, self.pipe_receiver, send_image_init_done))
        self.send_image_threads.append(process)
        process.start()

        self.logger.info("Waiting for send image init...")
        send_image_init = send_image_init_done.wait(20.0)
        if not send_image_init:
            self.logger.error("Send image init timeout")
            return False

        return True

    
    async def display(self):
        self.display_process = threading.Thread(target=self._displayer, args=())
        self.display_process.start()
        return True


    def close(self):
        self.logger.info("Stopping...")

        self.stop_event.set()

        # Stop distrubutor
        self.distributor.stop()
        if self.worker and self.worker.is_alive():
            self.logger.info("Stopping distribute worker...")
            self.worker.join()
            self.logger.info("Distribute worker stopped")

        # Stop displayer 
        if self.display_process and self.display_process.is_alive():
            self.logger.info("Stopping displayer worker...")
            # stop sender
            self.pipe_receiver.send(None)

            # stop displayer
            self.display_process.join()
            self.logger.info("Displayer worker stopped...")

        # Stop all the threads
        # TODO: timeout send_images
        for t in self.send_image_threads:
            t.join()
            self.logger.info("Processes stopped")

        self.logger.info("Stopped")


    def __send_images(self, port, pipe, init_ok: threading.Event):
        try:
            self.logger.info("Connecting to YOLO remote server")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.server_address, port))
            self.logger.info("Connected to YOLO remote server")
            init_ok.set()
            while True:
                image_bytes = pipe.recv()
                print("received")
                if image_bytes is None:
                    break
                client_socket.sendall(image_bytes)
                client_socket.send(b"IMAGE_COMPLETE")
                print("sent")

                result_bytes = b''
                while True:
                    chunk = client_socket.recv(1024)
                    if not chunk:
                        break
                    result_bytes += chunk
                    if result_bytes.endswith(b"ARRAY_COMPLETE"):
                        result_bytes = result_bytes[:-14]
                        break
                print("received 2")
                if result_bytes:
                    image = Image.open(io.BytesIO(result_bytes))
                    result = np.array(image)
                    img_shape = result.shape
                    if len(img_shape) == 2:
                        result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
                    pipe.send(result)
                    print("sent 2")
        except Exception as err:
            self.logger.error(f"Error in sending image: {type(err)=} {err}")
        finally:
            client_socket.close()
            self.logger.info("Connection to YOLO remote server closed")


    
    def _displayer(self):
        '''
        Receive YOLO result from the server, and put them in the queue
        '''
        # index = 0
        pipe = self.pipe_distributor
        queue = self.queue
        while True:
            if pipe.poll():
                result = pipe.recv()
                if result is None:
                    break
                if queue:
                    queue.put_nowait(result)
                    continue
            # TODO: seperate display function
        #     cv2.imshow(f'Image {index}', result)
        #     index += 1
        #     if cv2.waitKey(int(1000/fps*0.9)) & 0xFF == ord('q'):
        #         out_pipe.send(None)
        #         break
        # cv2.destroyAllWindows()