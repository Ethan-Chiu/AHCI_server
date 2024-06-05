import logging
from websockets.legacy.server import WebSocketServerProtocol

class Handler:
    def __init__(self, path: str, handler):
        self.path = path
        self.handler = handler
        self.clients = set()

    async def __call__(self, websocket: WebSocketServerProtocol, path: str) -> bool:
        if path == self.path:
            logging.info(f"Client connected: {path}")
            self.clients.add(websocket)
            try:
                await self.handler(websocket, self.clients)
                return True
            except:
                logging.warn("Client disconnected")
            finally:
                self.clients.remove(websocket)
        return False