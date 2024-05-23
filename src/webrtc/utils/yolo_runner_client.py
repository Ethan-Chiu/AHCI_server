import io
import socket
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Pipe
from PIL import Image
import threading
from .data_source import Distributor


class YoloRunnerClient:
    def __init__(self, queue: mp.Queue, cam_queue:mp.Queue, server_ip, port):
        self.queue = queue
        self.cam_queue = cam_queue

        self.server_address = server_ip
        self.base_port = port

        self.pipe_distributor, self.pipe_receiver = Pipe()
        self.display_pipe_parent, self.display_pipe = Pipe()

        self.processes = []
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
        port = self.base_port
        process = threading.Thread(target=self._send_images, args=(port, self.pipe_receiver, ))
        self.processes.append(process)
        process.start()
        
        self.worker = threading.Thread(target=self.distributor.start)
        self.worker.start()

    
    async def display(self):
        self.display_process = threading.Thread(target=self._displayer, args=())
        self.display_process.start()


    def close(self):
        print("Stopping YoloRunnerClient")            

        # Stop distrubutor
        self.stop_event.set()
        self.distributor.stop()
        self.worker.join()
        print("YoloRunnerClient Distrubutor stopped")

        # Stop displayer 
        if self.display_process and self.display_process.is_alive():
            print("Display process alive")
            # stop sender
            self.pipe_receiver.send(None)

            # stop displayer
            self.display_process.join()
            print("YoloRunnerClient Displayer stopped")

        # Stop all the processses
        for process in self.processes:
            process.join()
            print("YoloRunnerClient Process stopped")

        print("YoloRunnerClient stopped")


    def _send_images(self, port, pipe):
        try:
            print("Connecting to YOLO remote server")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.server_address, port))
            print("Connected to YOLO remote server")
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
            print("Error in sending image")
            print(f"Consumer error: Unexpected {err=}, {type(err)=}")
        finally:
            client_socket.close()
            print("Connection to YOLO remote server closed")


    
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
                    queue.put(result)
                    continue
            # TODO: seperate display function
        #     cv2.imshow(f'Image {index}', result)
        #     index += 1
        #     if cv2.waitKey(int(1000/fps*0.9)) & 0xFF == ord('q'):
        #         out_pipe.send(None)
        #         break
        # cv2.destroyAllWindows()