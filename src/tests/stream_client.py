import asyncio
import websockets
from websockets.exceptions import ConnectionClosed
import cv2
import base64
import json
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SERVER_URL = "ws://24.199.95.99:8000/ws/video-feed" # put here the URL of the server

async def send_frame(websocket, frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    await websocket.send(json.dumps({
        "type": "frame",
        "data": frame_base64
    }))

async def main():
    cap = cv2.VideoCapture(0)
    
    async for websocket in websockets.connect(SERVER_URL):
        try:
            logger.info("Connected to server")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break

                frame = cv2.resize(frame, (640, 480))
                await send_frame(websocket, frame)

                detection_results = await websocket.recv()
                results = json.loads(detection_results)

                for detection in results:
                    box = detection['box']
                    x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
                    label = detection['name']
                    conf = detection['confidence']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imshow('YOLO Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit")
                    return

        except ConnectionClosed:
            logger.info("Connection closed, attempting to reconnect...")
            continue
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(f"Error details: {str(e)}")
            await asyncio.sleep(5)  # Wait before attempting to reconnect

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")