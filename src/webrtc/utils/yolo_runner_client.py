import io
import socket
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from PIL import Image
import time
import threading


class YoloRunnerClient:
    def __init__(self, queue: mp.Queue, server_ip, server_num):
        self.queue = queue

        self.server_address = server_ip
        self.server_num = server_num
        self.base_port = 13761
        self.fps = 15

        self.pipes = [Pipe() for _ in range(self.server_num)]
        self.display_pipe = Pipe()
        self.processes = []
        self.display_process = None
        self.worker = None
        self.stop_event = mp.Event()


    def connect(self):
        processes = []
        for i in range(self.server_num):
            port = self.base_port + i
            process = Process(target=self._send_images, args=(port, self.pipes[i][1]))
            processes.append(process)
            process.start()
        
        self.worker = threading.Thread(target=self._distribute)
        self.worker.start()

    
    def display(self):
        self.display_process = Process(target=self._displayer, args=(self.display_pipe[1]))
        self.display_process.start()


    def close(self):
        print("Stopping YoloRunnerClient")
        # stop distrubutor
        self.stop_event.set()
        self.worker.join()
        print("YoloRunnerClient Distrubutor stopped")

        # stop sender
        for pipe in self.pipes:
            pipe[0].send(None)
        time.sleep(1)

        # stop displayer
        self.display_process.join()
        print("YoloRunnerClient Displayer stopped")

        for pipe in self.pipes:
            if pipe[0].poll():
                pipe[0].recv()

        for process in self.processes:
            process.join()
            print("YoloRunnerClient Process stopped")

        print("YoloRunnerClient stopped")


    def _distribute(self):
        print("Start connecting camera")
        cap = cv2.VideoCapture(0)
        print("Camera connected")
        index = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
            image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
            self.pipes[index % self.server_num][0].send(image_bytes)
            if self.display_pipe[0].poll(1/self.fps):
                if self.display_pipe[0].recv() is None:
                    break
            index += 1


    def _send_images(self, port, pipe):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_address, port))
        print("Yolo client connected")
        try:
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
                image = Image.open(io.BytesIO(result_bytes))
                result = np.array(image)[:, :, :3]
                pipe.send(result)
                print("sent 2")
        finally:
            client_socket.close()


    def _displayer(self, out_pipe):
        index = 0
        while True:
            result = self.pipes[index % self.server_num][0].recv()
            if self.queue:
                self.queue.put(result)
                continue
            cv2.imshow(f'Image {index}', result)
            index += 1
            if cv2.waitKey(int(1000/self.fps*0.9)) & 0xFF == ord('q'):
                out_pipe.send(None)
                break
        cv2.destroyAllWindows()
