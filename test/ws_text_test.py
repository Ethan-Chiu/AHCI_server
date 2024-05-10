import asyncio
import websockets

async def connect_to_websocket():
    uri = "ws://192.168.137.1:8080/SimpleDataChannelService"
    async with websockets.connect(uri) as websocket:
        while True:
            message = input("Enter a message to send (type 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            await websocket.send(message)
            print("Message sent: ", message)
            response = await websocket.recv()
            print("Received response:", response)

asyncio.get_event_loop().run_until_complete(connect_to_websocket())

