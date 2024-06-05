import asyncio
import socket
import logging
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from handler import Handler
from logger.logger_util import get_logger

class SimpleDataChannelServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None
        self.handlers: list[Handler] = []
        self.logger = get_logger("WsServer", log_level=logging.DEBUG)

    def add_handler(self, handler: Handler):
        self.handlers.append(handler)

    async def __handle_client(self, websocket: WebSocketServerProtocol, path: str):
        for handler in self.handlers:
            if await handler(websocket, path):
                return
        else:
            self.logger.warning(f"Received connection to unknown path: '{path}'")
            await websocket.close()

    async def start(self):
        self.server = await websockets.serve(self.__handle_client, self.host, self.port)
        self.logger.info(f"Server started at ws://{self.host}:{self.port}")

    async def stop(self):
        self.logger.info("Server shutting down...")
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def handle_message(websocket: WebSocketServerProtocol, clients, logger: logging.Logger):
    async for message in websocket:
        logger.debug(f"Received message: {message[:30]}...")
        logger.debug(f"Clients: {clients}")
        for client in clients:
            if client != websocket:  # Skip the sender
                logger.debug(f"Sending message to {client}")
                await client.send(message)

async def handle_pose_data(websocket: WebSocketServerProtocol, clients, logger: logging.Logger):
    async for data in websocket:
        for client in clients:
            if client != websocket:
                logger.debug(f"Recieve data and send to {client}")
                await client.send(data)
            

async def main():
    # logging.basicConfig(level=logging.DEBUG)

    # Get the local IPv4 address
    host = socket.gethostbyname(socket.gethostname())
    port = 8080
    
    server = SimpleDataChannelServer(host, port)

    video_handler = Handler("/video", handle_message)
    pose_handler = Handler("/posedata", handle_pose_data)
    
    video_handler.set_log_level(logging.DEBUG)
    pose_handler.set_log_level(logging.DEBUG)

    server.add_handler(video_handler)
    server.add_handler(pose_handler)
    
    await server.start()

    # Keep the server running until interrupted
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
