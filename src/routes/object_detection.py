from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from src.models.object_detection import ObjectDetectionModel
import logging
import json
import torch

logging.basicConfig(filename='api_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

router = APIRouter()

def load_model(model_name: str):
    if torch.cuda.is_available():
        logging.info("GPU is available. Loading model to GPU.")
        return ObjectDetectionModel.get_model(model_name).to('cuda')
    else:
        logging.info("GPU is not available. Loading model to CPU.")
        return ObjectDetectionModel.get_model(model_name)

@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Query("yolov10n.pt", description="Name of the YOLO model to use")
):
    model = load_model(model_name)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = np.array(image)
    
    # If GPU is available, move image to GPU
    if torch.cuda.is_available():
        image_np = torch.from_numpy(image_np).to('cuda')
    
    results = model.track(image_np, stream=True)
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
    
    # Log the response
    logging.info(f"API Response: {json.dumps(detections, indent=2)}")
    
    return JSONResponse(content=detections)