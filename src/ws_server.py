import asyncio
import socket
from websockets.legacy.server import serve, WebSocketServerProtocol


class SimpleDataChannelServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None
        self.clients = set()
        self.pose_clients = set()

    async def handle_client(self, websocket, path):
        if path == "/video":
            print("Video: Client connected ", websocket)
            self.clients.add(websocket)
            try:
                async for message in websocket:
                    await self.handle_message(websocket, message)
            except:
                print("Client disconnected")
            finally:
                self.clients.remove(websocket)
        elif path == '/posedata':
            print("PoseData: Client connected ", websocket)
            self.pose_clients.add(websocket)
            try:
                async for posedata in websocket:
                    await self.handle_pose_data(websocket, posedata)
            except:
                print('connection error')
            finally:
                self.pose_clients.remove(websocket)
        else:
            print(f"Received connection to unknown path: '{path}'")
            await websocket.close()

    async def handle_message(self, websocket: WebSocketServerProtocol, message):
        # Handle received message here
        print(f"Received message: {message}")

        print("Clients", self.clients)
        for client in self.clients:
            if client != websocket:  # Skip the sender
                print(f"Sending message to {client}")
                await client.send(message)

    async def handle_pose_data(self, websocket: WebSocketServerProtocol, posedata):
        for client in self.pose_clients:
            if client != websocket:
                print(f"Recieve data and send to {client}")
                await client.send(posedata)


    async def start(self):
        self.server = await serve(self.handle_client, self.host, self.port)
        print(f"Server started at ws://{self.host}:{self.port}/video")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def main():
    # Get the local IPv4 address
    host = socket.gethostbyname(socket.gethostname())
    port = 8080

    server = SimpleDataChannelServer(host, port)
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
