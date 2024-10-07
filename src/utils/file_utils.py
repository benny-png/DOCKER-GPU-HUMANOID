import os
import shutil
from fastapi import UploadFile

def get_next_file_number(directory: str, prefix: str) -> int:
    existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f[len(prefix):].split('.')[0].isdigit()]
    if not existing_files:
        return 1
    numbers = [int(f[len(prefix):].split('.')[0]) for f in existing_files]
    return max(numbers) + 1

def save_uploaded_file(file: UploadFile, directory: str, prefix: str) -> str:
    file_extension = os.path.splitext(file.filename)[1]
    next_number = get_next_file_number(directory, prefix)
    filename = f"{prefix}{next_number}{file_extension}"
    file_path = os.path.join(directory, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path