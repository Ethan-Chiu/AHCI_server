import asyncio
import websockets
import random
import json
import socket

async def send_random_data(uri):
    async with websockets.connect(uri) as websocket:
        while True:
            # Create some random data to send
            data = {
                "value": random.randint(0, 100),
            }
            # Convert the data to a JSON string
            message = json.dumps(data)
            print(f"Sending: {message}")
            # Send the data to the server
            await websocket.send(message)
            # Wait for a bit before sending the next message
            await asyncio.sleep(1)


host = socket.gethostbyname(socket.gethostname())
websocket_uri = f"ws://{host}:8080/posedata"

# Start the asyncio event loop and run the client
asyncio.get_event_loop().run_until_complete(send_random_data(websocket_uri))
