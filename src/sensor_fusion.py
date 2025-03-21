import cv2
import numpy as np
import os

def refine_detections_with_contours(image, detections):
    """
    Refine each detection using contour detection.
    For each detection, crop the region of interest (ROI), apply edge detection,
    find the largest contour, and update the bounding box.
    """
    refined_detections = []
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        # Crop the ROI from the image
        roi = image[y1:y2, x1:x2]
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        # Find contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Choose the largest contour as the refined object boundary
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Adjust coordinates to match the original image
            refined_bbox = [x1 + x, y1 + y, x1 + x + w, y1 + y + h]
            det['refined_bbox'] = refined_bbox
        else:
            det['refined_bbox'] = bbox  # Keep original if no contour found
        refined_detections.append(det)
    return refined_detections

if __name__ == '__main__':
    # Updated path to use image_00 instead of image_02:
    image_path = os.path.join(
        os.getcwd(), 
        'data', 
        '2011_09_26_drive_0048_sync', 
        '2011_09_26', 
        '2011_09_26_drive_0048_sync', 
        'image_00',  # Changed from image_02 to image_00
        'data', 
        '0000000000.png'  # You can test with frame 0 here
    )
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found at:", image_path)
        exit(1)
    
    # Simulated detection results (from your data preprocessing step)
    detections = [
        {'bbox': [72.04, 208.04, 338.84, 333.11], 'confidence': 0.90, 'class': 'car'},
        {'bbox': [684.18, 190.07, 793.35, 270.82], 'confidence': 0.88, 'class': 'car'}
    ]
    
    refined = refine_detections_with_contours(image, detections)
    print("Refined Detections:")
    for det in refined:
        print(det)
