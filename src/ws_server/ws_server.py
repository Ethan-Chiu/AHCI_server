import asyncio
import socket
import logging
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from handler import Handler


class SimpleDataChannelServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None
        self.handlers: list[Handler] = []

    def add_handler(self, handler: Handler):
        self.handlers.append(handler)

    async def __handle_client(self, websocket: WebSocketServerProtocol, path: str):
        for handler in self.handlers:
            if await handler(websocket, path):
                return
        else:
            logging.warn(f"Received connection to unknown path: '{path}'")
            await websocket.close()

    async def start(self):
        self.server = await websockets.serve(self.__handle_client, self.host, self.port)
        print(f"Server started at ws://{self.host}:{self.port}")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def handle_message(websocket: WebSocketServerProtocol, clients):
    async for message in websocket:
        logging.debug(f"Received message: {message[:30]}...")
        logging.debug(f"Clients: {clients}")
        for client in clients:
            if client != websocket:  # Skip the sender
                logging.debug(f"Sending message to {client}")
                await client.send(message)

async def handle_pose_data(websocket: WebSocketServerProtocol, clients):
    async for data in websocket:
        for client in clients:
            if client != websocket:
                logging.debug(f"Recieve data and send to {client}")
                await client.send(data)
            

async def main():
    logging.basicConfig(level=logging.DEBUG)

    # Get the local IPv4 address
    host = socket.gethostbyname(socket.gethostname())
    port = 8080
    
    server = SimpleDataChannelServer(host, port)

    video_handler = Handler("/video", handle_message)
    pose_handler = Handler("/posedata", handle_pose_data)
    
    server.add_handler(video_handler)
    server.add_handler(pose_handler)
    
    await server.start()

    # Keep the server running until interrupted
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        print("Server shutting down...")
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
