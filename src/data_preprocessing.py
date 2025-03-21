import os
import cv2
import torch
import numpy as np

# Update the IMAGE_FOLDER path without '..'
IMAGE_FOLDER = os.path.join(
    os.getcwd(),
    'data',
    '2011_09_26_drive_0048_sync',
    '2011_09_26',
    '2011_09_26_drive_0048_sync',
    'image_02',
    'data'
)

def load_image(frame_number):
    """
    Load an image given a frame number.
    KITTI images are numbered with 10-digit filenames.
    """
    filename = f"{frame_number:010d}.png"  # Use 10-digit numbering
    img_path = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return image

def run_object_detection(image):
    """
    Run YOLOv5 object detection on the image using PyTorch Hub.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    return results

def simulate_radar_data():
    """
    Simulate radar data for testing.
    For now, assign random distance and relative speed.
    """
    radar_data = {
        'distance': np.random.uniform(5, 50),  # distance in meters
        'relative_speed': np.random.uniform(-5, 5)  # relative speed in m/s
    }
    return radar_data

def preprocess_frame(frame_number):
    """
    Preprocess a single frame: load image, run object detection, simulate radar data, and combine results.
    """
    image = load_image(frame_number)
    results = run_object_detection(image)
    
    # Extract detection results (bounding boxes, class, confidence)
    detections = []
    for det in results.xyxy[0].tolist():
        x1, y1, x2, y2, conf, cls = det
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class': results.names[int(cls)]
        })
    
    radar_data = simulate_radar_data()
    
    processed_data = {
        'frame_number': frame_number,
        'detections': detections,
        'radar': radar_data
    }
    
    return processed_data

if __name__ == '__main__':
    frame_number = 10
    try:
        processed_data = preprocess_frame(frame_number)
        print("Processed Data for Frame", frame_number)
        print(processed_data)
    except FileNotFoundError as e:
        print(e)
