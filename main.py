from src.data_preprocessing import preprocess_frame
from src.sensor_fusion import refine_detections_with_contours
from src.decision_making import collision_avoidance
import cv2
import os

def main():
    frame_number = 10
    # Step 2: Preprocess the frame to get detections and simulated radar data
    fused_data = preprocess_frame(frame_number)
    
    # Load the original image for refinement
    image_path = os.path.join(
        os.getcwd(),
        'data',
        '2011_09_26_drive_0048_sync',
        '2011_09_26',
        '2011_09_26_drive_0048_sync',
        'image_02',
        'data',
        f"{frame_number:010d}.png"
    )
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found for refinement")
        return
    
    # Step 3: Refine detections using contour detection
    refined_detections = refine_detections_with_contours(image, fused_data['detections'])
    fused_data['detections'] = refined_detections
    
    # Step 4: Apply decision making for collision avoidance
    decisions = collision_avoidance(fused_data)
    
    # Output the results
    print("Fused Data with Refined Detections:")
    print(fused_data)
    print("Decision Actions:")
    for d in decisions:
        print(d)

if __name__ == '__main__':
    main()
