from fastapi import APIRouter, File, UploadFile, Query
from PIL import Image
import io
import numpy as np
from src.models.object_detection import ObjectDetectionModel

router = APIRouter()

@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = Query("yolov10x.pt", description="Name of the YOLO model to use")
):
    model = ObjectDetectionModel.get_model(model_name)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = np.array(image)
    
    results = model(image_np, stream=True)
    
    return next(results).tojson()