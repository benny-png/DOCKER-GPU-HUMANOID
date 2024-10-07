from fastapi import APIRouter, File, UploadFile, Query, Form
from fastapi.responses import JSONResponse
import os
import tempfile
import numpy as np
from src.models.face_recognition import FaceRecognitionModel
from src.utils.file_utils import save_uploaded_file

router = APIRouter()

@router.post("/find_faces")
async def find_faces(
    file: UploadFile = File(...),
    db_path: str = Query(..., description="Path to the database of face images"),
    model_name: str = Query("VGG-Face", description="Name of the face recognition model to use"),
    detector_backend: str = Query("retinaface", description="Name of the face detector backend to use")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name

    try:
        result = FaceRecognitionModel.find_faces(
            img_path=temp_file_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend
        )
        
        serializable_result = []
        for df in result:
            serializable_df = df.to_dict(orient="records")
            for record in serializable_df:
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        record[key] = value.tolist()
            serializable_result.append(serializable_df)
        
        return JSONResponse(content={"result": serializable_result})
    
    finally:
        os.unlink(temp_file_path)

@router.post("/add_face")
async def add_face(
    file: UploadFile = File(...),
    person_name: str = Form(...),
    db_path: str = Query(..., description="Path to the database of face images")
):
    person_dir = os.path.join(db_path, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    file_path = save_uploaded_file(file, person_dir, person_name)
    
    return JSONResponse(content={"message": f"Face added for {person_name}", "file_path": file_path})