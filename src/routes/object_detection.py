from fastapi import APIRouter, File, UploadFile, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
from src.models.object_detection import ObjectDetectionModel
import logging
import json
import torch
import base64
import cv2

logging.basicConfig(filename='api_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

router = APIRouter()

class ModelLoadError(Exception):
    pass

class ImageProcessingError(Exception):
    pass

class YOLOProcessingError(Exception):
    pass



def load_model(model_name: str):
    try:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Loading model {model_name} to {device}")
        model = ObjectDetectionModel.get_model(model_name)
        if model is None:
            raise ModelLoadError(f"Failed to load model: {model_name}")
        return model, device
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        raise ModelLoadError(f"Failed to load model: {str(e)}")



def process_image(image):
    try:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise ImageProcessingError(f"Failed to process image: {str(e)}")




@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Query("yolov10n.pt", description="Name of the YOLO model to use")
):
    try:
        model, device = load_model(model_name)
        
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
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
    except ModelLoadError as e:
        logging.error(f"Model loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except ImageProcessingError as e:
        logging.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in detect_objects: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")




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
                    if image_np is None:
                        raise ImageProcessingError("Failed to decode image")
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
                    await websocket.send_text(json.dumps({"error": f"Unknown message type: {message['type']}"}))

            except WebSocketDisconnect:
                logging.info("WebSocket disconnected by client")
                break
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {str(e)}")
                await websocket.send_text(json.dumps({"error": f"Invalid JSON: {str(e)}"}))
            except KeyError as e:
                logging.error(f"KeyError in message processing: {str(e)}")
                await websocket.send_text(json.dumps({"error": f"Missing key in message: {str(e)}"}))
            except base64.binascii.Error as e:
                logging.error(f"Base64 decode error: {str(e)}")
                await websocket.send_text(json.dumps({"error": f"Invalid base64 encoding: {str(e)}"}))
            except ImageProcessingError as e:
                logging.error(f"Image processing error: {str(e)}")
                await websocket.send_text(json.dumps({"error": f"Image processing failed: {str(e)}"}))
            except YOLOProcessingError as e:
                logging.error(f"YOLO processing error: {str(e)}")
                await websocket.send_text(json.dumps({"error": f"YOLO processing failed: {str(e)}"}))
            except Exception as e:
                logging.error(f"Unexpected error in WebSocket loop: {str(e)}", exc_info=True)
                await websocket.send_text(json.dumps({"error": f"Unexpected error: {str(e)}"}))

    except ModelLoadError as e:
        logging.error(f"Model loading error: {str(e)}")
        await websocket.send_text(json.dumps({"error": f"Failed to load model: {str(e)}"}))
    except Exception as e:
        logging.error(f"Unexpected error in WebSocket setup: {str(e)}", exc_info=True)
        await websocket.send_text(json.dumps({"error": f"Unexpected error in setup: {str(e)}"}))
    finally:
        logging.info("WebSocket connection closed")
        await websocket.close()