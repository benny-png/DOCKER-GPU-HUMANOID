from fastapi import APIRouter, File, UploadFile, Query, Form, HTTPException, Path
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import os
import tempfile
import numpy as np
from src.models.face_recognition import FaceRecognitionModel
from src.utils.file_utils import save_uploaded_file
import asyncio
import functools
import torch
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
operation_semaphore = asyncio.Semaphore(20)

class FaceRecognitionError(Exception):
    pass

class DatabaseError(Exception):
    pass

class ImageProcessingError(Exception):
    pass




def run_face_recognition(img_path, db_path, model_name, detector_backend):
    try:
        return FaceRecognitionModel.find_faces(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend
        )
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}", exc_info=True)
        raise FaceRecognitionError(f"Face recognition failed: {str(e)}")




async def async_run_face_recognition(img_path, db_path, model_name, detector_backend):
    async with operation_semaphore:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, 
                functools.partial(run_face_recognition, img_path, db_path, model_name, detector_backend)
            )
        except FaceRecognitionError as e:
            logger.error(f"Error in async face recognition: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in async face recognition: {str(e)}", exc_info=True)
            raise FaceRecognitionError(f"Unexpected error in face recognition: {str(e)}")



@router.post("/find_faces", description="Detect and recognize faces in an uploaded image")
async def find_faces(
    file: UploadFile = File(..., description="Image file to analyze"),
    db_path: str = Query(..., description="Path to the database of face images"),
    model_name: str = Query("VGG-Face", description="Name of the face recognition model to use"),
    detector_backend: str = Query("retinaface", description="Name of the face detector backend to use")
):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        result = await async_run_face_recognition(temp_file_path, db_path, model_name, detector_backend)
        
        if not result:
            logger.info("No faces detected in the image.")
            return JSONResponse(content={"message": "No faces detected in the image."}, status_code=200)
        
        serializable_result = []
        for df in result:
            serializable_df = df.to_dict(orient="records")
            for record in serializable_df:
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        record[key] = value.tolist()
            serializable_result.append(serializable_df)
        
        return JSONResponse(content={"result": serializable_result})
    
    except FileNotFoundError:
        logger.error(f"Database path not found: {db_path}")
        raise HTTPException(status_code=404, detail=f"Database path not found: {db_path}")
    except ValueError as ve:
        logger.error(f"Invalid input: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except FaceRecognitionError as fre:
        logger.error(f"Face recognition error: {str(fre)}")
        raise HTTPException(status_code=500, detail=str(fre))
    except Exception as e:
        logger.error(f"Unexpected error in find_faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)




@router.post("/add_face", description="Add a new face image to the database")
async def add_face(
    file: UploadFile = File(..., description="Face image to add"),
    person_name: str = Form(..., description="Name of the person"),
    db_path: str = Query(..., description="Path to the database of face images")
):
    try:
        async with operation_semaphore:
            person_dir = os.path.join(db_path, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            file_path = await save_uploaded_file(file, person_dir, person_name)
        
        return JSONResponse(content={"message": f"Face added for {person_name}", "file_path": file_path})
    except FileNotFoundError:
        logger.error(f"Database path not found: {db_path}")
        raise HTTPException(status_code=404, detail=f"Database path not found: {db_path}")
    except PermissionError:
        logger.error(f"Permission denied when writing to {db_path}")
        raise HTTPException(status_code=403, detail=f"Permission denied when writing to database: {db_path}")
    except Exception as e:
        logger.error(f"Unexpected error in add_face: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")





@router.get("/list_people", response_model=List[str], description="List all people in the face database")
async def list_people(db_path: str = Query(..., description="Path to the database of face images")):
    try:
        people = [name for name in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, name))]
        return people
    except FileNotFoundError:
        logger.error(f"Database path not found: {db_path}")
        raise HTTPException(status_code=404, detail=f"Database path not found: {db_path}")
    except Exception as e:
        logger.error(f"Error listing people: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while listing people: {str(e)}")




@router.get("/person/{person_name}", response_model=List[str], description="Get all images for a specific person")
async def get_person_images(
    person_name: str = Path(..., description="Name of the person"),
    db_path: str = Query(..., description="Path to the database of face images")
):
    person_dir = os.path.join(db_path, person_name)
    try:
        if not os.path.isdir(person_dir):
            raise FileNotFoundError(f"Person '{person_name}' not found")
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return images
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting images for {person_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving images for {person_name}: {str(e)}")



@router.get("/image/{person_name}/{image_name}", description="Retrieve a specific image file")
async def get_image(
    person_name: str = Path(..., description="Name of the person"),
    image_name: str = Path(..., description="Name of the image file"),
    db_path: str = Query(..., description="Path to the database of face images")
):
    image_path = os.path.join(db_path, person_name, image_name)
    if not os.path.isfile(image_path):
        logger.error(f"Image not found: {image_path}")
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
    return FileResponse(image_path)




@router.put("/update_person/{old_name}", description="Update a person's name in the database")
async def update_person_name(
    old_name: str = Path(..., description="Current name of the person"),
    new_name: str = Form(..., description="New name for the person"),
    db_path: str = Query(..., description="Path to the database of face images")
):
    old_path = os.path.join(db_path, old_name)
    new_path = os.path.join(db_path, new_name)
    try:
        if not os.path.isdir(old_path):
            raise FileNotFoundError(f"Person '{old_name}' not found")
        if os.path.exists(new_path):
            raise ValueError(f"Person '{new_name}' already exists")
        os.rename(old_path, new_path)
        return JSONResponse(content={"message": f"Successfully renamed '{old_name}' to '{new_name}'"})
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating person name from {old_name} to {new_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while updating the person's name: {str(e)}")





@router.delete("/delete_person/{person_name}", description="Delete a person and all their images from the database")
async def delete_person(
    person_name: str = Path(..., description="Name of the person to delete"),
    db_path: str = Query(..., description="Path to the database of face images")
):
    person_dir = os.path.join(db_path, person_name)
    try:
        if not os.path.isdir(person_dir):
            raise FileNotFoundError(f"Person '{person_name}' not found")
        shutil.rmtree(person_dir)
        return JSONResponse(content={"message": f"Successfully deleted '{person_name}' and all associated images"})
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting person {person_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting {person_name}: {str(e)}")





@router.delete("/delete_image/{person_name}/{image_name}", description="Delete a specific image for a person")
async def delete_image(
    person_name: str = Path(..., description="Name of the person"),
    image_name: str = Path(..., description="Name of the image to delete"),
    db_path: str = Query(..., description="Path to the database of face images")
):
    image_path = os.path.join(db_path, person_name, image_name)
    try:
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found")
        os.remove(image_path)
        return JSONResponse(content={"message": f"Successfully deleted image '{image_name}' for '{person_name}'"})
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting image {image_name} for {person_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the image: {str(e)}")




@router.get("/system_info", description="Get information about the system's GPU availability")
async def get_system_info():
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
        else:
            device_name = "CPU"
            device_count = 1

        return JSONResponse(content={
            "cuda_available": cuda_available,
            "device_name": device_name,
            "device_count": device_count
        })
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving system information: {str(e)}")