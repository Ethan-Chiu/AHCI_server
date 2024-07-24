import time
import asyncio
import numpy as np
import time
import logging
import threading

from logger.logger_util import get_logger

from utils.data_source import PoseDataSource, CameraDataSource

class ProducerConsumer:
    def __init__(self, logger: logging.Logger, latest_data, stop_event, out_pipe, cam_queue, fps, cam_source):
        self.latest_data = latest_data
        self.stop_event = stop_event
        self.out_pipe = out_pipe
        self.cam_queue = cam_queue
        self.fps = fps
        self.cam_source = cam_source
        self.logger = get_logger("ProducerConsumer")

        self.pose_data_source = PoseDataSource()
        self.cam_data_source = CameraDataSource(cam_source)

    async def start(self):
        self.logger.info("Starting...")
        try: 
            self.logger.info("Starting data sources")
            pose_data_init, cam_data_init = await asyncio.gather(
                self.pose_data_source.start(),
                self.cam_data_source.start()
            )
            return pose_data_init and cam_data_init
        except Exception as e: 
            self.logger.error(f"Failed to start. {e}")
            return False

    async def run(self):
        self.logger.info("Running...")
        try:
            start_time = time.time()
            while not self.stop_event.is_set():
                # Produce data
                self.logger.debug("Producing data")
                pose_data = await self.pose_data_source.get_data()
                self.logger.debug("Pose data")
                cam_data, frame_data = self.cam_data_source.get_data()
                self.logger.debug("Cam data")

                if not pose_data or not cam_data:
                    await asyncio.sleep(1)
                    continue

                parsed_pose_data = pose_data.split(b'torch')
                handhead = b"handhead" + parsed_pose_data[0]

                if len(parsed_pose_data) != 1:
                    self.logger.info("torch activated")
                    handhead += b"torch"
                    handhead += parsed_pose_data[1]
                
                return_bytes = cam_data + handhead

                self.logger.debug("Data prepared")
                self.cam_queue.put_nowait(frame_data)
                self.out_pipe.send(return_bytes)
                self.logger.debug("Send pipe")

                sleep_time = max(0, 1/self.fps - (time.time() - start_time))
                time.sleep(sleep_time)
                self.logger.debug("Waited")
                start_time = time.time()
        except asyncio.exceptions.CancelledError:
            self.logger.warning("Run cancelled")
        except Exception as err:
            self.logger.error(f"Run error: Unexpected {err=}, {type(err)=}")
        finally:
            await self.__close()

    async def __close(self):
        self.logger.info("Closing...")
        await self.pose_data_source.end()
        self.cam_data_source.end()
        while self.out_pipe.poll():
            self.out_pipe.recv()
        self.logger.info("Closed")


class Distributor:
    def __init__(self, producerconsumer_args):
        self.latest_data = [b"", b"", np.ones((640, 480, 3)), -1]  # Use a list to store the latest data (mutable type)
        self.stop_event = asyncio.Event()
        self.logger = get_logger("Distributor")

        self.loop = None
        self.producerconsumer = ProducerConsumer(self.logger, self.latest_data, self.stop_event, **producerconsumer_args)

        self.producerconsumer_task = None

    def run(self, init_done: threading.Event, error_flag: threading.Event):
        self.logger.info("Running...")
        try: 
            self.loop = asyncio.new_event_loop()
            start_ok = self.loop.run_until_complete(self.__start())
            if not start_ok:
                self.logger.error("Failed to start")
                init_done.set()
                error_flag.set()
                return 
            init_done.set()
            self.logger.info("Started")
            self.loop.run_until_complete(self.__run())
        except asyncio.exceptions.CancelledError:
            self.logger.warning("Distributor cancelled")
        except Exception as e:
            self.logger.error(f"Error running: {e}")

    async def __start(self):
        self.logger.info("Starting...")
        producerconsumer_init = await self.producerconsumer.start()
        if not producerconsumer_init:
            self.logger.error("Failed to init ProducerConsumer")
            return False
        self.logger.info("ProducerConsumer initialized")
        return True

    async def __run(self):
        self.producerconsumer_task = asyncio.create_task(self.producerconsumer.run())
        await self.producerconsumer_task

    def stop(self):
        self.logger.info("Stopping...")
        self.stop_event.set()

        if self.producerconsumer_task and not self.producerconsumer_task.done():
            self.logger.info("Canceling ProducerConsumer...")
            self.producerconsumer_task.cancel()
            while not self.producerconsumer_task.cancelled() and not self.producerconsumer_task.done():
                time.sleep(0.1)
            self.producerconsumer_task = None
            self.logger.info("ProducerConsumer cancelled")

        self.logger.info("Stopped")
