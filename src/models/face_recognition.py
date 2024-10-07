
from deepface import DeepFace

class FaceRecognitionModel:
    @staticmethod
    def find_faces(img_path: str, db_path: str, model_name: str, detector_backend: str):
        return DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend
        )