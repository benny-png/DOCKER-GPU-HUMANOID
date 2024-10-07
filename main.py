from fastapi import FastAPI
from src.routes import object_detection, face_recognition

app = FastAPI()

app.include_router(object_detection.router)
app.include_router(face_recognition.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)