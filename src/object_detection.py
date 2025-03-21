# src/object_detection_preprocessing.py

import cv2
import torch
import os
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def run_object_detection(image):
    # Load YOLOv5 model from PyTorch Hub (first time it downloads the model)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    return results

def simulate_radar_data(detections):
    # For each detection, simulate radar information: distance (m) and relative speed (m/s)
    fused_data = []
    for *box, conf, cls in detections.xyxy[0].tolist():
        # Simulated values (for demo purposes, random values or fixed assumptions)
        radar_info = {
            'distance': np.random.uniform(5, 20),  # example: distance between 5 and 20 meters
            'relative_speed': np.random.uniform(-3, 3)  # relative speed between -3 and 3 m/s
        }
        fused_data.append({
            'class': detections.names[int(cls)],
            'bbox': box,
            'confidence': conf,
            **radar_info
        })
    return fused_data

def main():
    # Path to one image from the synced+rectified folder
    image_folder = 'data/images/synced_rectified'
    image_file = os.path.join(image_folder, '000000.png')  # Adjust filename as needed

    # Load image
    image = load_image(image_file)

    # Run object detection
    results = run_object_detection(image)
    print("Detection Results:")
    print(results.xyxy[0])  # Print detection tensor

    # Simulate and fuse radar data
    fused_results = simulate_radar_data(results)
    print("\nFused Data (with simulated radar info):")
    for obj in fused_results:
        print(obj)

    # Optionally, draw the detections on the image and save for documentation
    annotated_image = np.squeeze(results.render())  # Render returns a list of images
    output_path = 'data/images/annotated_output.png'
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"\nAnnotated image saved at: {output_path}")

if __name__ == '__main__':
    main()

