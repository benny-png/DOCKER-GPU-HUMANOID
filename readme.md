## API Endpoints

### 1. Object Detection

**Endpoint**: `/detect`

**Method**: POST

**Parameters**:
- `file`: The image file to perform object detection on (form-data)
- `model_name`: Name of the YOLO model to use (query parameter, default: "yolov10n.pt")

**Example Usage**:

```python
import requests

def detect_objects(image_path, model_name="yolov10n.pt"):
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

**Response Format**:

The API returns a list of detections, where each detection is a dictionary containing information about the detected object, including class name, confidence, bounding box coordinates, and track ID.

### 2. Face Recognition

#### 2.1 Find Faces

**Endpoint**: `/find_faces`

**Method**: POST

**Parameters**:
- `file`: The image file to analyze (form-data)
- `db_path`: Path to the database of face images (query)
- `model_name`: Name of the face recognition model to use (query, default: "VGG-Face")
- `detector_backend`: Name of the face detector backend to use (query, default: "retinaface")

#### 2.2 Add Face

**Endpoint**: `/add_face`

**Method**: POST

**Parameters**:
- `file`: Face image to add (form-data)
- `person_name`: Name of the person (form)
- `db_path`: Path to the database of face images (query)

#### 2.3 List People

**Endpoint**: `/list_people`

**Method**: GET

**Parameters**:
- `db_path`: Path to the database of face images (query)

#### 2.4 Get Person Images

**Endpoint**: `/person/{person_name}`

**Method**: GET

**Parameters**:
- `person_name`: Name of the person (path)
- `db_path`: Path to the database of face images (query)

#### 2.5 Get Image

**Endpoint**: `/image/{person_name}/{image_name}`

**Method**: GET

**Parameters**:
- `person_name`: Name of the person (path)
- `image_name`: Name of the image file (path)
- `db_path`: Path to the database of face images (query)

#### 2.6 Update Person Name

**Endpoint**: `/update_person/{old_name}`

**Method**: PUT

**Parameters**:
- `old_name`: Current name of the person (path)
- `new_name`: New name for the person (form)
- `db_path`: Path to the database of face images (query)

#### 2.7 Delete Person

**Endpoint**: `/delete_person/{person_name}`

**Method**: DELETE

**Parameters**:
- `person_name`: Name of the person to delete (path)
- `db_path`: Path to the database of face images (query)

#### 2.8 Delete Image

**Endpoint**: `/delete_image/{person_name}/{image_name}`

**Method**: DELETE

**Parameters**:
- `person_name`: Name of the person (path)
- `image_name`: Name of the image to delete (path)
- `db_path`: Path to the database of face images (query)

### 3. System Information

**Endpoint**: `/system_info`

**Method**: GET

Retrieves information about the system's GPU availability.

### 4. WebSocket for Real-time Video Processing

**Endpoint**: `/ws/video-feed`

**Method**: WebSocket

Allows real-time video processing for object detection. Clients can send video frames and receive detection results in real-time.

For detailed usage of these endpoints, please refer to the API documentation available at `http://localhost:8000/docs` when the server is running.