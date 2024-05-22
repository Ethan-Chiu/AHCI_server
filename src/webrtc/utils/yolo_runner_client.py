import io
import socket
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from PIL import Image
import threading
from .data_source import Distributor


class YoloRunnerClient:
    def __init__(self, queue: mp.Queue, server_ip, port):
        self.queue = queue

        self.server_address = server_ip
        self.base_port = port
        self.fps = 15

        self.pipe_distributor, self.pipe_receiver = Pipe()
        self.display_pipe_parent, self.display_pipe = Pipe()

        self.processes = []
        self.display_process: threading.Thread | None = None
        self.worker = None
        self.stop_event = mp.Event()

        self.distributor = Distributor(self.pipe_distributor)


    def connect(self):
        port = self.base_port
        process = Process(target=self._send_images, args=(port, self.pipe_receiver, ))
        self.processes.append(process)
        process.start()
        
        self.worker = threading.Thread(target=self.distributor.start)
        self.worker.start()

    
    def display(self):
        self.display_process = threading.Thread(target=self._displayer, args=())
        self.display_process.start()


    def close(self):
        print("Stopping YoloRunnerClient")            

        # Stop distrubutor
        self.stop_event.set()
        while self.pipe_receiver.poll():
            print("Drain pipe")
            self.pipe_receiver.recv()
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


    # def _distribute(self):
    #     '''
    #     Send camera input to YOLO remote server
    #     '''
    #     print("Start connecting camera")
    #     cap = cv2.VideoCapture(0)
    #     print("Camera connected")
    #     index = 0
    #     while not self.stop_event.is_set():
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("No camera input!")
    #             break
    #         resized = cv2.resize(frame, (640, int(640*frame.shape[0]/frame.shape[1])))
    #         image_bytes = cv2.imencode('.jpg', resized)[1].tobytes()
    #         self.pipe_distributor.send(image_bytes)

    #         if self.display_pipe_parent.poll(1/self.fps):
    #             if self.display_pipe_parent.recv() is None:
    #                 break

    #         index += 1


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
                image = Image.open(io.BytesIO(result_bytes))
                result = np.array(image)[:, :, :3]
                pipe.send(result)
                print("sent 2")
        except:
            print("Error in sending image")
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
            if self.pipe_distributor.poll():
                result = pipe.recv()
                if result == None:
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