import asyncio
import socket
from websockets.legacy.server import serve, WebSocketServerProtocol


class SimpleDataChannelServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = None
        self.clients = set()

    async def handle_client(self, websocket, path):
        if path == "/video":
            print("Client connected ", websocket)
            self.clients.add(websocket)  # Add client to the set of connected clients
            try:
                async for message in websocket:
                    await self.handle_message(websocket, message)
            except:
                print("Client disconnected")
            finally:
                self.clients.remove(websocket)
        else:
            print(f"Received connection to unknown path: {path}")
            await websocket.close()

    async def handle_message(self, websocket: WebSocketServerProtocol, message):
        # Handle received message here
        print(f"Received message: {message}")

        print("Clients", self.clients)
        for client in self.clients:
            if client != websocket:  # Skip the sender
                print(f"Sending message to {client}")
                await client.send(message)

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
