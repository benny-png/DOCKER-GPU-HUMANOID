# Face Recognition and Object Detection API

## Project Structure

```
project_root/
│
├── src/
│   ├── database/
│   │   └── (face images organized by person)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── object_detection.py
│   │   └── face_recognition.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── object_detection.py
│   │   └── face_recognition.py
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py
│
├── main.py
└── requirements.txt
```

- `src/database/`: Contains face images organized by person (each person has their own folder).
- `src/models/`: Contains the core logic for object detection and face recognition.
- `src/routes/`: Defines the API endpoints.
- `src/utils/`: Contains utility functions, such as file handling.
- `main.py`: The entry point of the application.
- `requirements.txt`: Lists all the Python dependencies.

## Installation

1. Clone the repository or download the project files.
2. Navigate to the project root directory.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Server

From the project root directory, run:

```bash
python main.py
```

The server will start running on `http://localhost:8000` by default.

## API Endpoints

### 1. Object Detection

**Endpoint**: `/detect`

**Method**: POST

**Parameters**:
- `file`: The image file to perform object detection on (form-data)
- `model_name`: Name of the YOLO model to use (query parameter, default: "yolov8n.pt")

**Example Usage**:

```python
import requests

def detect_objects(image_path, model_name="yolov8n.pt"):
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(
            f"http://localhost:8000/detect?model_name={model_name}",
            files=files
        )
    return response.json()

results = detect_objects("/path/to/your/image.jpg")
print(results)
```

**Recreating Original Results**:

The JSON response from this endpoint is already in a format very close to the original YOLO results. To work with it in a way similar to the original YOLO output:

```python
import json
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops

class CustomResults(Results):
    def __init__(self, json_data):
        self.__dict__.update(json_data)
        self.boxes = ops.Boxes(self.boxes.numpy(), self.boxes.shape)

# Assuming 'results' is the JSON response from the API
json_data = json.loads(results)
custom_results = CustomResults(json_data)

# Now you can use custom_results similar to original YOLO results
for box in custom_results.boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}")
```

### 2. Face Recognition

**Endpoint**: `/find_faces`

**Method**: POST

**Parameters**:
- `file`: The image file to perform face recognition on (form-data)
- `db_path`: Path to the database of face images (query parameter)
- `model_name`: Name of the face recognition model to use (query parameter, default: "VGG-Face")
- `detector_backend`: Name of the face detector backend to use (query parameter, default: "retinaface")

**Example Usage**:

```python
import requests

def find_faces(image_path, db_path, model_name="VGG-Face", detector_backend="retinaface"):
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(
            f"http://localhost:8000/find_faces?db_path={db_path}&model_name={model_name}&detector_backend={detector_backend}",
            files=files
        )
    return response.json()

results = find_faces("/path/to/your/image.jpg", "/path/to/your/database")
print(results)
```

**Recreating Original Results**:

The JSON response from this endpoint contains serialized pandas DataFrames. To work with them as original DeepFace results:

```python
import pandas as pd
import numpy as np

def deserialize_deepface_result(json_result):
    deserialized_result = []
    for df_json in json_result['result']:
        df = pd.DataFrame(df_json)
        for column in df.columns:
            if isinstance(df[column].iloc[0], list):
                df[column] = df[column].apply(np.array)
        deserialized_result.append(df)
    return deserialized_result

# Assuming 'results' is the JSON response from the API
deserialized_results = deserialize_deepface_result(results)

# Now you can work with deserialized_results as if they were the original DeepFace output
for df in deserialized_results:
    print(df.head())
```

### 3. Add Face to Database

**Endpoint**: `/add_face`

**Method**: POST

**Parameters**:
- `file`: The image file of the face to add (form-data)
- `person_name`: Name of the person (form-data)
- `db_path`: Path to the database of face images (query parameter)

**Example Usage**:

```python
import requests

def add_face_to_database(image_path, person_name, db_path):
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        data = {"person_name": person_name}
        response = requests.post(
            f"http://localhost:8000/add_face?db_path={db_path}",
            files=files,
            data=data
        )
    return response.json()

result = add_face_to_database("/path/to/face/image.jpg", "John Doe", "/path/to/your/database")
print(result)
```

## Notes

- Ensure that the `db_path` used in face recognition and adding faces points to the `src/database/` directory in your project structure.
- The face images in the database are organized by person name. Each person has their own folder containing their images.
- Face images are saved with names like "PersonName1.jpg", "PersonName2.jpg", etc., continuing from the highest existing number.
- The object detection model uses YOLO from the Ultralytics library.
- The face recognition functionality uses the DeepFace library.

For any issues or feature requests, please contact us the project maintainers.