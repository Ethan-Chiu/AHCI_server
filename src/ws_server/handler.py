import logging
from websockets.legacy.server import WebSocketServerProtocol
from logger.logger_util import get_logger

class Handler:
    def __init__(self, path: str, handler):
        self.path = path
        self.handler = handler
        self.clients = set()
        self.logger = get_logger(f"Handler {path}")

    def set_log_level(self, level):
        self.logger.setLevel(level)
        
    async def __call__(self, websocket: WebSocketServerProtocol, path: str) -> bool:
        if path == self.path:
            self.logger.info(f"Client connected: {path}")
            self.clients.add(websocket)
            try:
                await self.handler(websocket, self.clients, self.logger)
                return True
            except Exception as e:
                self.logger.error(f"Error: {e}")
                self.logger.warning("Client disconnected")
            finally:
                self.clients.remove(websocket)
        return False