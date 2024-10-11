from fastapi import APIRouter, File, UploadFile, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from src.models.object_detection import ObjectDetectionModel
import logging
import json
import torch
import base64
import cv2

logging.basicConfig(filename='api_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

router = APIRouter()

def load_model(model_name: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading model to {device}")
    return ObjectDetectionModel.get_model(model_name), device

def process_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return np.array(image)

@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Query("yolov10n.pt", description="Name of the YOLO model to use")
):
    model, device = load_model(model_name)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = process_image(image)
    
    results = model.track(image_np, stream=True, device=device)
    result = next(results)
    
    detections = []
    for box in result.boxes:
        detection = {
            "name": result.names[int(box.cls)],
            "class": int(box.cls),
            "confidence": float(box.conf),
            "box": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            }
        }
        if box.id is not None:
            detection["track_id"] = int(box.id)
        detections.append(detection)
    
    logging.info(f"API Response: {json.dumps(detections, indent=2)}")
    
    return JSONResponse(content=detections)

@router.websocket("/ws/video-feed")
async def websocket_endpoint(
    websocket: WebSocket,
    model_name: str = Query("yolov10n.pt", description="Name of the YOLO model to use")
):
    await websocket.accept()
    logging.info(f"WebSocket connection accepted. Model: {model_name}")
    
    try:
        model, device = load_model(model_name)

        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "frame":
                    logging.debug("Received full frame, processing")
                    img_data = base64.b64decode(message["data"])
                    nparr = np.frombuffer(img_data, np.uint8)
                    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    logging.debug(f"Image processed. Shape: {image_np.shape}")

                    results = model.track(image_np, stream=True, device=device)
                    result = next(results)
                    logging.debug("YOLO processing completed")

                    detections = []
                    for box in result.boxes:
                        detection = {
                            "name": result.names[int(box.cls)],
                            "class": int(box.cls),
                            "confidence": float(box.conf),
                            "box": {
                                "x1": float(box.xyxy[0][0]),
                                "y1": float(box.xyxy[0][1]),
                                "x2": float(box.xyxy[0][2]),
                                "y2": float(box.xyxy[0][3])
                            }
                        }
                        if box.id is not None:
                            detection["track_id"] = int(box.id)
                        detections.append(detection)

                    logging.debug(f"Sending response with {len(detections)} detections")
                    await websocket.send_text(json.dumps(detections))
                else:
                    logging.warning(f"Received unknown message type: {message['type']}")

            except WebSocketDisconnect:
                logging.info("WebSocket disconnected by client")
                break
            except Exception as e:
                logging.error(f"Error in WebSocket loop: {e}")
                logging.error(f"Error details: {str(e)}")
                await websocket.send_text(json.dumps({"error": str(e)}))
                break

    except Exception as e:
        logging.error(f"Error in WebSocket setup: {e}")
        logging.error(f"Error details: {str(e)}")
    finally:
        logging.info("WebSocket connection closed")