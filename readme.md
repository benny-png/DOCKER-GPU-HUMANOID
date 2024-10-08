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

The API returns a list of detections, where each detection is a dictionary containing the following information:

```json
[
  {
    "name": "person",
    "class": 0,
    "confidence": 0.96824,
    "box": {
      "x1": 0.64642,
      "y1": 71.78952,
      "x2": 334.61133,
      "y2": 478.8522
    },
    "track_id": 1
  },
  ...
]
```

- `name`: The class name of the detected object
- `class`: The class ID of the detected object
- `confidence`: The confidence score of the detection
- `box`: The bounding box coordinates of the detection (x1, y1, x2, y2)
- `track_id`: A unique identifier for tracking the object across frames

**Working with the Results**:

You can process the results directly from the JSON response. Here's an example of how to work with the detections:

```python
import json

# Assuming 'response' is the API response
detections = json.loads(response.text)

for detection in detections:
    print(f"Detected {detection['name']} with confidence {detection['confidence']:.2f}")
    print(f"Bounding box: {detection['box']}")
    print(f"Track ID: {detection['track_id']}")
    print("---")
```

**Visualizing Results**:

To visualize the results, you can use a library like OpenCV to draw bounding boxes and labels on the image. Here's a simple example:

```python
import cv2
import numpy as np

def draw_boxes(image_path, detections):
    image = cv2.imread(image_path)
    for detection in detections:
        box = detection['box']
        x1, y1 = int(box['x1']), int(box['y1'])
        x2, y2 = int(box['x2']), int(box['y2'])
        label = f"{detection['name']} {detection['confidence']:.2f}"
        
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Use the function
image_path = "/path/to/your/image.jpg"
detections = detect_objects(image_path)
draw_boxes(image_path, detections)
```

This will display the image with bounding boxes and labels for each detected object.

**Notes**:
- The `yolov10n.pt` model is used by default. You can specify different YOLO models by changing the `model_name` parameter.
- The API handles tracking of objects across frames, providing a `track_id` for each detection.
- The bounding box coordinates are returned in the format (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the box.
- Confidence scores range from 0 to 1, with higher values indicating greater confidence in the detection.

For any issues or feature requests related to object detection, please contact us the project maintainers.
