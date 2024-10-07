from ultralytics import YOLO

class ObjectDetectionModel:
    _instances = {}

    @classmethod
    def get_model(cls, model_name: str):
        if model_name not in cls._instances:
            cls._instances[model_name] = YOLO(model_name)
        return cls._instances[model_name]